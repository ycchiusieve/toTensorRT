# Orin handoff: Qwen3.8-27B INT8 SQ + MTP split-prefill engine

Target: Jetson AGX Orin 64 GB, JetPack 7.2 (CUDA 13.x, TRT 10.16 aarch64), batch 1,
input/KV 65536, FP16 KV, MTP=3, split prefill at 4096. Built **on the Orin** —
engines don't move between GPUs.

## 1. What to copy to the Orin

Hardware-neutral, copy as-is:

| From (Thor) | Size | Note |
|---|---|---|
| `modelopt-test/out/Qwen3.8-27B-edgellm-int8_sq-mtp-onnx/` (`llm/`, `mtp_draft/`, `visual/`) | ~32 GB | the ONNX export; `trtedge export` output is device-independent |
| `.claude/skills/totrt/patches/split-prefill-profile-v0.11.1.patch` | 132 lines | the split-prefill patch |
| `.claude/skills/totrt/scripts/` (`build_bench.sh`, `profile_engine_layers.py`, `compare_engine_profiles.py`) | KB | profiling / bench tools |

Do **not** copy: `build-thor*` (built for sm_110), any `*.engine` (Thor-only).
Reclone or copy the Edge-LLM **source** v0.11.1 (`TensorRT-Edge-LLM/`, minus build dirs).

## 2. Build Edge-LLM on the Orin (with the patch)

```bash
git clone <repo> TensorRT-Edge-LLM && cd TensorRT-Edge-LLM
git checkout v0.11.1
git apply /path/to/split-prefill-profile-v0.11.1.patch   # 3 files: builder/llmBuilder.{h,cpp}, runtime/exec/engineExecutor.cpp
cmake -B build-orin -DEMBEDDED_TARGET=jetson-orin -DTRT_PACKAGE_DIR=/usr
cmake --build build-orin -j"$(nproc)"
```

`jetson-orin` sets `CMAKE_CUDA_ARCHITECTURES 87` and, with CUDA ≥ 13, the
`sbsa-linux` toolkit dir (checked in `cmake/aarch64_linux_toolchain.cmake:80`);
it also adds `TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT` (CMakeLists.txt:161). No other
flags were needed on Thor (`-DEMBEDDED_TARGET=jetson-thor -DTRT_PACKAGE_DIR=/usr`).

## 3. Build the engine

`build_bench.sh` picks it up through the environment:

```bash
export EDGELLM_BUILD=$PWD/build-orin
EDGELLM_SHORT_PREFILL_TOKENS=4096 \
  scripts/build_bench.sh --onnx <onnx dir> --engine <engine dir> --mtp 3 \
  --bench longctx_4096_input.json longctx_32768_input.json gsm8k100_input.json
```

Expect the log line `Added short prefill profile 2: tokens [1, 4096], opt 4096`.
Engine ≈ 27 GB, so keep ≥ 120 GB free on the NVMe (ONNX + engine + build temp).
64 GB RAM: stop the desktop and every other container before building, add NVMe
swap (see SKILL.md Step 4 note). If the build or a 64K-context run OOMs, drop to
`--input-len 32768 --kv 32768` first before touching anything else.

## 4. Run and measure

The engine **must** run with the patched `build-orin/examples/llm/llm_inference`
(a stock runtime only uses profile 0 and silently loses the 4K speedup).
Bench the same inputs as Thor (`--spec-decode --spec-draft-top-k 1 --spec-draft-step 3
--spec-verify-size 4 --warmup 1 --dump-profile`) and score gsm8k-100 — Thor reference:
96% split vs 95% stock.

`profile_engine_layers.py <engine> <plugin.so> out.json --prefill 4096 --prefill-profile 2`
then `compare_engine_profiles.py` against the no-split engine built the same day.

## 5. What to expect (and re-measure) on Orin

- **Different kernels.** sm_87 Ampere picks Ampere INT8 (IMMA) / Myelin sm80 kernels;
  none of Thor's `cutlass3x_sm100_*` or the fused `FcSilu` 256×256 kernels exist.
  The split *mechanism* is identical; which tactics win per shape is not portable.
- **Re-confirm the split point.** 4096 was Thor's measured break-even. On Orin, prefill
  is slower, so the short profile likely helps *more*; try `EDGELLM_SHORT_PREFILL_TOKENS=2048`
  and `8192` if the prompt mix is far from 4K (one extra build each, ~2× Thor build time).
- **Speed.** Decode ~25% slower than Thor (205 vs 273 GB/s bandwidth). INT8 SQ is the
  right format on Orin — it has no FP8/FP4 tensor cores; INT8 is its only native fast path.
- Accuracy should match the stock Orin engine within noise, same as on Thor.

Thor reference numbers (all of the above were measured there): SKILL.md "Reference"
table and Step 5b's stock-vs-split table.
