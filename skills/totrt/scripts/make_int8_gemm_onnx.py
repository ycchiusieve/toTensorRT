#!/usr/bin/env python3
"""One ONNX per Qwen3.8-27B linear shape: X fp16 [M,K] -> Q/DQ (per-tensor int8) @ DQ(int8 W, per-channel) -> Y fp16 [M,N].
This is the pattern Edge-LLM exports for W8A8 SmoothQuant, which TensorRT fuses into an INT8 GEMM.
Each model holds G independent copies of the weight (G * N * K >= 300 MB) applied to the same X, so the
timed weights can't sit in cache the way one reused weight would; divide the run time by G (written to
<name>.g). M is dynamic; build one engine per M.

    python3 make_int8_gemm_onnx.py <out dir>
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SHAPES = {  # name: (N, K)
    "gdn_in_proj_qkv": (10240, 5120), "gdn_in_proj_z": (6144, 5120), "gdn_out_proj": (5120, 6144),
    "attn_q_proj": (12288, 5120), "attn_kv_proj": (1024, 5120), "attn_o_proj": (5120, 6144),
    "mlp_gate_up": (17408, 5120), "mlp_down": (5120, 17408),
}


def copies(n, k):
    return max(1, -(-300 * 2**20 // (n * k)))


def build(n, k):
    rng = np.random.default_rng(0)
    inits = [numpy_helper.from_array(np.array(0.02, np.float16), "x_scale"),
             numpy_helper.from_array(np.array(0, np.int8), "x_zp"),
             numpy_helper.from_array(np.zeros(n, np.int8), "w_zp")]
    nodes = [helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["xq"]),
             helper.make_node("DequantizeLinear", ["xq", "x_scale", "x_zp"], ["xdq"])]
    outs = []
    for i in range(copies(n, k)):
        inits += [numpy_helper.from_array(rng.integers(-127, 128, size=(n, k), dtype=np.int8), f"w_q{i}"),
                  numpy_helper.from_array((rng.random(n, np.float32) * 1e-3 + 1e-4).astype(np.float16), f"w_scale{i}")]
        nodes += [helper.make_node("DequantizeLinear", [f"w_q{i}", f"w_scale{i}", "w_zp"], [f"wdq{i}"], axis=0),
                  helper.make_node("Transpose", [f"wdq{i}"], [f"wt{i}"], perm=[1, 0]),
                  helper.make_node("MatMul", ["xdq", f"wt{i}"], [f"y{i}"])]
        outs.append(helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT16, ["M", n]))
    g = helper.make_graph(nodes, "int8_gemm", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["M", k])],
                          outs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for name, (n, k) in SHAPES.items():
        onnx.save(build(n, k), out / f"{name}.onnx", save_as_external_data=True, location=f"{name}.onnx.data")
        (out / f"{name}.g").write_text(str(copies(n, k)))
        print(name, n, k, copies(n, k), flush=True)
