#!/usr/bin/env python3
"""Per-layer GPU time of an Edge-LLM Qwen3.8 spec_base engine (TensorRT IProfiler), grouped by kernel kind.

Feeds one step the way the runtime does (shape extents select the execution phase):
  verify  : profile 1, 4 tokens, execution_phase_marker len 5 (kSpecTargetVerify), chain-tree mask
  prefill : profile 0, N tokens, execution_phase_marker len 1 (kContextPrefill)
Cache/state tensors are zero-filled; present_* outputs alias their past_* inputs. Timings only, not outputs.
Build the engine with `llm_build --profiling-detailed` to get the tactic (kernel) names.

    python3 profile_engine_layers.py <spec_base.engine> <plugin.so> <out.json> [--past 100] [--prefill 4096]
"""
import argparse
import ctypes
import json
import re
from collections import defaultdict

import numpy as np
import tensorrt as trt

cudart = ctypes.CDLL("libcudart.so.13")
H2D = 1


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: cuda error {rc}")


def dalloc(nbytes):
    p = ctypes.c_void_p()
    ck(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(nbytes, 256))), "cudaMalloc")
    ck(cudart.cudaMemset(p, 0, ctypes.c_size_t(max(nbytes, 256))), "cudaMemset")
    return p.value


def upload(ptr, arr):
    arr = np.ascontiguousarray(arr)
    ck(cudart.cudaMemcpy(ctypes.c_void_p(ptr), arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(arr.nbytes), H2D),
       "cudaMemcpy")


NP = {trt.DataType.HALF: np.float16, trt.DataType.FLOAT: np.float32, trt.DataType.INT32: np.int32,
      trt.DataType.INT64: np.int64, trt.DataType.INT8: np.int8, trt.DataType.BF16: np.uint16,
      trt.DataType.BOOL: np.bool_, trt.DataType.FP8: np.uint8}


class Prof(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.t = defaultdict(list)

    def report_layer_time(self, name, ms):
        self.t[name].append(ms)


def kind(layer):
    tac = layer.get("TacticName", "") or ""
    name = layer.get("Name", "")
    lt = layer.get("LayerType", "")
    if lt == "PluginV2" or lt == "PluginV3" or "Plugin" in name:
        return "plugin:" + (re.sub(r"_\d+$", "", layer.get("Metadata", "") or name)[:40])
    if re.search(r"gemm|gemv|tensorop|_Fc_|cutlass", tac, re.I) or re.search(r"linear|gemm|matmul", name, re.I):
        return "gemm"
    return "other"


def step_inputs(eng, phase, n, past):
    """name -> (shape, ndarray or None for zeros)."""
    pos = np.arange(past, past + n, dtype=np.int32)
    total = past + n
    d = {
        "inputs_embeds": ((n, 5120), (np.random.default_rng(0).standard_normal((n, 5120)) * 0.05).astype(np.float16)),
        "rope_rotary_cos_sin": ((n, 64), np.concatenate([np.cos(pos[:, None] * 1e-3 * np.arange(32)),
                                                        np.sin(pos[:, None] * 1e-3 * np.arange(32))], 1).astype(np.float32)),
        "positions": ((n,), pos),
        "query_start_offsets": ((2,), np.array([0, n], np.int32)),
        "query_lengths": ((1,), np.array([n], np.int32)),
        "past_lengths": ((1,), np.array([past], np.int32)),
        "attention_sequence_lengths": ((1,), np.array([total], np.int32)),
        "state_indices": ((1,), np.array([0], np.int32)),
        "execution_phase_marker": (((5 if phase == "verify" else 1),), np.zeros(5 if phase == "verify" else 1, np.int32)),
        "context_sequence_count_carrier": (((0 if phase == "verify" else 1),), np.zeros(1, np.int32)),
        "logits_indices": (((n if phase == "verify" else 1),), (np.arange(n) if phase == "verify" else np.array([n - 1])).astype(np.int64)),
        "attention_position_ids": ((n,), pos),
        "packed_attention_mask": ((n, 1), np.array([(1 << (i + 1)) - 1 for i in range(n)] if phase == "verify"
                                                  else np.zeros(n), np.int32).reshape(n, 1)),
    }
    pt = eng.get_tensor_shape("kv_page_table")
    pages = int(pt[2])
    d["kv_page_table"] = (tuple(pt), np.stack([np.arange(pages), np.arange(pages) + pages]).astype(np.int32)[None])
    return d


def run(eng, plugin_layers, phase, n, past, iters, prefill_profile=0):
    ctx = eng.create_execution_context()
    ctx.set_optimization_profile_async(1 if phase == "verify" else prefill_profile, 0)
    feeds = step_inputs(eng, phase, n, past)
    bufs, names = {}, [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    for nm in names:
        if eng.get_tensor_mode(nm) != trt.TensorIOMode.INPUT:
            continue
        shape, arr = feeds.get(nm, (tuple(eng.get_tensor_shape(nm)), None))
        if -1 in shape:
            raise RuntimeError(f"no feed for dynamic input {nm}")
        ctx.set_input_shape(nm, shape)
        nb = int(np.prod(shape)) * np.dtype(NP[eng.get_tensor_dtype(nm)]).itemsize
        bufs[nm] = dalloc(nb)
        if arr is not None:
            upload(bufs[nm], arr.astype(NP[eng.get_tensor_dtype(nm)]))
        ctx.set_tensor_address(nm, bufs[nm])
    for nm in names:
        if eng.get_tensor_mode(nm) != trt.TensorIOMode.OUTPUT:
            continue
        alias = next((a for a in (nm.replace("present_", "past_", 1), nm.replace("present_", "", 1)) if a in bufs), None)
        if alias:
            ptr = bufs[alias]
        else:
            shape = tuple(ctx.get_tensor_shape(nm))
            ptr = dalloc(int(np.prod(shape)) * np.dtype(NP[eng.get_tensor_dtype(nm)]).itemsize)
        ctx.set_tensor_address(nm, ptr)
    stream = ctypes.c_void_p()
    ck(cudart.cudaStreamCreate(ctypes.byref(stream)), "stream")

    def once():
        if not ctx.execute_async_v3(stream.value):
            raise RuntimeError("enqueue failed")
        ck(cudart.cudaStreamSynchronize(stream), "sync")

    once(); once()
    ev0, ev1 = ctypes.c_void_p(), ctypes.c_void_p()
    cudart.cudaEventCreate(ctypes.byref(ev0)); cudart.cudaEventCreate(ctypes.byref(ev1))
    ts = []
    for _ in range(iters):
        cudart.cudaEventRecord(ev0, stream)
        ctx.execute_async_v3(stream.value)
        cudart.cudaEventRecord(ev1, stream)
        ck(cudart.cudaEventSynchronize(ev1), "evsync")
        ms = ctypes.c_float()
        cudart.cudaEventElapsedTime(ctypes.byref(ms), ev0, ev1)
        ts.append(ms.value)
    prof = Prof()
    ctx.profiler = prof
    for _ in range(iters):
        once()
    del ctx
    for p in bufs.values():
        cudart.cudaFree(ctypes.c_void_p(p))
    return float(np.median(ts)), {k: float(np.median(v)) for k, v in prof.t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine"); ap.add_argument("plugin"); ap.add_argument("out")
    ap.add_argument("--past", type=int, default=100)
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prefill-profile", type=int, default=0, help="2 = short-prefill profile of a split engine")
    a = ap.parse_args()
    ctypes.CDLL(a.plugin)
    lg = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(lg, "")
    eng = trt.Runtime(lg).deserialize_cuda_engine(open(a.engine, "rb").read())
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = {l["Name"]: l for l in info["Layers"] if isinstance(l, dict)}
    result = {}
    for phase, n in (("verify", 4), ("prefill", a.prefill)):
        total, per = run(eng, layers, phase, n, a.past if phase == "verify" else 0, a.iters, a.prefill_profile)
        groups, gemm_tac = defaultdict(float), defaultdict(lambda: [0, 0.0])
        for nm, ms in per.items():
            l = layers.get(nm, {"Name": nm})
            g = kind(l)
            groups[g] += ms
            if g == "gemm":
                t = gemm_tac[l.get("TacticName", "?") or "?"]
                t[0] += 1; t[1] += ms
        prof_sum = sum(per.values())
        print(f"\n== {phase} ({n} tokens, past {a.past if phase == 'verify' else 0}): engine step {total:.2f} ms "
              f"(no profiler), per-layer sum {prof_sum:.2f} ms over {len(per)} layers")
        for g, ms in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {g:48s} {ms:9.2f} ms  {100 * ms / prof_sum:5.1f}%")
        print("  GEMM tactics:")
        for tac, (c, ms) in sorted(gemm_tac.items(), key=lambda x: -x[1][1])[:15]:
            print(f"    {c:4d} x {ms / c * 1000:8.1f} us = {ms:8.2f} ms  {tac[:110]}")
        result[phase] = {"tokens": n, "step_ms": total, "layers": per, "groups": groups,
                         "gemm_tactics": {k: v for k, v in gemm_tac.items()}}
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
