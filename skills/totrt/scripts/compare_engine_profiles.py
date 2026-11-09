#!/usr/bin/env python3
"""Compare two spec_base engines' per-layer profiles (profile_engine_layers.py output + engine layer info):
GEMM time and chosen tactic per (K, N) shape, for the verify and prefill phases.

    python3 compare_engine_profiles.py <A.engine> <A_profile.json> <B.engine> <B_profile.json> <plugin.so>
"""
import collections
import ctypes
import json
import re
import sys

import tensorrt as trt


def layer_info(engine, lg):
    e = trt.Runtime(lg).deserialize_cuda_engine(open(engine, "rb").read())
    info = json.loads(e.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    return {l["Name"]: l for l in info["Layers"] if isinstance(l, dict)}


def by_shape(layers, prof):
    g = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    other = 0.0
    for n, ms in prof["layers"].items():
        l = layers.get(n, {})
        tac = l.get("TacticName") or ""
        if l.get("LayerType") == "gemm" or re.search("gemm|gemv", tac):
            key = (l["Inputs"][0]["Dimensions"][-1], l["Outputs"][0]["Dimensions"][-1])
            e = g[key]; e[0] += 1; e[1] += ms
            e[2][re.sub(r"0x[0-9a-f]+_|_valpha.*|_relu.*|_bias.*", "", tac).replace("cutlass3x_sm100_tensorop_", "")[:48]] += 1
        else:
            other += ms
    return g, other


def main():
    ea, pa, eb, pb, plugin = sys.argv[1:6]
    ctypes.CDLL(plugin)
    lg = trt.Logger(trt.Logger.ERROR); trt.init_libnvinfer_plugins(lg, "")
    A, B = layer_info(ea, lg), json.load(open(pa))
    C, D = layer_info(eb, lg), json.load(open(pb))
    for ph in ("verify", "prefill"):
        ga, oa = by_shape(A, B[ph]); gb, ob = by_shape(C, D[ph])
        print(f"\n== {ph}: step A {B[ph]['step_ms']:.1f} ms, B {D[ph]['step_ms']:.1f} ms; non-GEMM A {oa:.1f} ms, B {ob:.1f} ms")
        for key in sorted(set(ga) | set(gb), key=lambda k: -max(ga.get(k, [0, 0])[1], gb.get(k, [0, 0])[1])):
            a, b = ga.get(key, [0, 0.0, {}]), gb.get(key, [0, 0.0, {}])
            print(f"  K{key[0]:>6} N{key[1]:>7} x{max(a[0], b[0]):3d}  A {a[1]:8.2f} ms  B {b[1]:8.2f} ms  "
                  f"({b[1] - a[1]:+7.2f})\n      A: {dict(a[2])}\n      B: {dict(b[2])}")


if __name__ == "__main__":
    main()
