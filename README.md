# toTensorRT

Claude Code skill for turning LLM / VLM / VLA checkpoints into TensorRT engines.

| skill | what it does |
|---|---|
| [`totrt`](skills/totrt/SKILL.md) | Converts a checkpoint into TensorRT engines with NVIDIA TensorRT Edge-LLM (`trtedge export` → `llm_build` / `visual_build`) on Jetson (Thor, Orin) or a discrete GPU, then tunes which GEMM kernel each module uses, including separate kernels for short (<4K) and long (≥4K) prefill. Covers vision and language towers as separate engines, MTP speculative decoding, per-layer profiling and batch-1 latency / memory measurement. |

It starts from a quantized checkpoint made by the `modelopt` skill
([model-quantization](https://github.com/ycchiusieve/model-quantization)), or from a BF16 model for an FP16 engine.

## Install
```bash
git clone git@github.com:ycchiusieve/toTensorRT.git
cp -r toTensorRT/skills/totrt ~/.claude/skills/        # all projects
# or: cp -r toTensorRT/skills/totrt <project>/.claude/skills/   # one project
```
Then in Claude Code ask e.g. "build a TensorRT engine from this checkpoint" or run `/totrt`.

## Layout
```
skills/totrt/
  SKILL.md                                   instructions Claude follows
  orin-handoff.md                            moving a Thor-tested build to a Jetson AGX Orin
  patches/split-prefill-profile-v0.11.1.patch  Edge-LLM v0.11.1: separate prefill profiles (<4K / >=4K)
  scripts/
    build_bench.sh                           build engines from a trtedge export, measure batch-1 latency and memory
    profile_engine_layers.py                 per-layer GPU time of an engine (TensorRT IProfiler), by kernel kind
    compare_engine_profiles.py               compare two engines' per-layer profiles: GEMM time and tactic per shape
    make_int8_gemm_onnx.py                   one ONNX per linear shape with the W8A8 SmoothQuant Q/DQ pattern
    trt_opt_shape_bench.sh                   effect of the prefill profile's opt shape on INT8 GEMM time
```
