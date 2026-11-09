---
name: totrt
description: Convert an LLM, VLM or VLA checkpoint into TensorRT engines with NVIDIA TensorRT Edge-LLM (trtedge export → llm_build / visual_build), on Jetson (Thor, Orin) or a discrete GPU, then tune which GEMM kernel each module uses, including separate kernels for short (<4K) and long (≥4K) prefill. Use when the user asks to build, convert to or optimize a TensorRT / TRT engine, mentions llm_build, trtexec, Edge-LLM, engine shapes, tactics or kernel selection, or hands over a quantized checkpoint from /modelopt (int8_sq, int4_awq, fp8, nvfp4) or an ONNX export. Covers vision and language towers as separate engines, MTP speculative decoding, per-layer profiling and batch-1 latency/memory measurement.
---

# TensorRT engines with TensorRT Edge-LLM

This skill turns a model into TensorRT engines and makes them fast. Quantization is `/modelopt`'s
job; this skill starts from its checkpoint (or from a BF16 model for an FP16 engine). Everything
here was run end to end on a Jetson AGX Thor with Qwen3.8-27B: a hybrid GatedDeltaNet +
full-attention VLM with an MTP head, INT8 SmoothQuant and INT4 AWQ, Edge-LLM **v0.11.1**,
TensorRT 10.16.2.

Why Edge-LLM and not TensorRT-LLM: TensorRT-LLM has no Orin / JetPack 7 build, and for GDN models
(Qwen3.5/3.8) only its PyTorch runtime works, with no engine path. Edge-LLM supports Thor, Orin,
DGX Spark and x86 GPUs, and handles GDN, MTP and separate vision towers.

Same rules as `/modelopt`:
- Python steps run in a Docker container, never in the host's Python. No `docker run --rm`.
- The Edge-LLM C++ runtime (`llm_build`, `llm_inference`) is a native build against the host's
  JetPack TensorRT, like `trtexec`. It lives in a source tree and isn't a package install.
- Downloads are authenticated; nothing is downloaded anonymously.
- Once inputs are confirmed, run end to end without asking "proceed?".

## Step 0: Confirm inputs (one round)

Check the platform first (`uname -m`, `/etc/nv_tegra_release`, `nvidia-smi`, `free -g`,
`nvpmodel -q`), then ask only for what the request leaves open:

- **Source**: a `/modelopt` checkpoint, an existing Edge-LLM ONNX export, or a BF16 model.
- **Target device.** The engine must be built **on** it. If that's another machine (e.g. an Orin
  with no SSH), the deliverable is the ONNX plus exact build commands, and the dev box only builds
  a validation engine.
- **Engine shapes**, fixed at build time:
  - max batch,
  - max input length,
  - KV-cache capacity,
  - image tokens per image / per request,
  - MTP on/off and draft count (3 tested).
  Give a memory estimate with the question (Step 3).
- **Typical prompt length.** This decides the kernel tuning in Step 5.
- **Host output dir.**

Precision limits by platform:
- **Orin (sm_87):** FP16 / INT8 / INT4 only. KV cache is FP16 only (FP8 KV needs SM89+).
- **Thor (sm_110):** also FP8 and NVFP4.

## Step 1: Export ONNX (container)

Use the `edgellm` container from `/modelopt`:
- image `vllm/vllm-openai:v0.27.1-ubuntu2404` (the wheel needs glibc 2.39);
- `pip install "tensorrt-edgellm[tools]==0.11.1" flash-linear-attention==0.6.0`.

```bash
docker exec -e HF_HUB_OFFLINE=1 edgellm bash -lc \
  "trtedge export /workspace/out/<ckpt> /workspace/out/<ckpt>-onnx [--mtp] --target-sm <87|110>"
docker exec edgellm chown -R $(id -u):$(id -g) /workspace/out/<ckpt>-onnx
```

- Output has one directory per tower, each holding its ONNX and external weights:
  - `llm/`: also gets the tokenizer, chat template and embedding.
  - `mtp_draft/`: only with `--mtp`.
  - `visual/`: the vision tower. A VLA's action head is exported the same way when Edge-LLM
    supports it; check with `trtedge export --help` and the repo's `examples/`.
- No shapes are fixed at export.
- `--target-sm` only matters for NVFP4. INT8 / INT4 ONNX works on any target.
- For an FP16 engine, export the BF16 checkpoint directly.
- A `/modelopt` checkpoint from `hf_ptq.py` is not an Edge-LLM checkpoint. For INT8 / INT4,
  quantize with `trtedge quantize llm` instead (see `/modelopt`), then export.

## Step 2: Build the Edge-LLM runtime on the target (host)

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git && cd TensorRT-Edge-LLM
git checkout v0.11.1 && git submodule update --init --recursive    # same version as the export
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=<jetson-thor|jetson-orin|igx-thor|gb10> -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL
make -j$(nproc)
```

- For an x86 GPU, drop the toolchain and target flags.
- Every build and run needs `EDGELLM_PLUGIN_PATH=<build>/libNvInfer_edgellm_plugin.so`.

## Step 3: Build and measure

```bash
EDGELLM_BUILD=<build> scripts/build_bench.sh --onnx <export> --engine <dir> \
  --batch 1 --input-len 65536 --kv 65536 --mtp 3 --visual \
  --bench gsm8k_b1.json longctx_4096.json longctx_32768.json
```

- The script builds the base, draft and visual engines.
- It logs build time and peak memory.
- It then runs `llm_inference --warmup 1 --dump-profile` on each input. That reports:
  - prefill time,
  - `Overall Tokens/Second (excluding base prefill)` (decode),
  - MTP accept length,
  - per-step verify / draft times,
  - `Peak GPU Memory`.

Memory, for the Step 0 question:

- **KV cache** = 2 × attention layers × KV heads × head_dim × 2 bytes per token.
  - Qwen3.8-27B: 64 KiB/token.
  - GDN layers add a fixed ~0.15 GB per sequence.
- **Prefill workspace** grows with max input length: ~20 GiB at 131072.
- **Qwen3.8-27B measured on Thor (runtime peak):**
  - INT8 + MTP, b1, 64K input / 64K KV: 46.3 GB.
  - INT4 + MTP, b1, 131072 / 131072: 56.3 GB.
  - INT8, batch 4, 131072: doesn't fit 64 GB.
- **The build peaks higher than the runtime:** +55.6 GB for INT8 + MTP 64K.
  - On a 64 GB Orin, stop the desktop and all containers before building, and add swap on NVMe.
  - If it still OOMs, lower max input length.

Accuracy:
- Use a separate **high-batch eval engine**, e.g. `--batch 16 --input-len 1024 --kv 2048`
  (`llm_inference` caps a batch at 16).
- Never benchmark accuracy one request at a time on the deployment engine.

Known v0.11.1 bugs:
- `llm_bench` spec / prefill modes crash on GDN spec-base engines. Use `llm_inference --dump-profile`.
- `llm_inference` caps each content item at 128 KB, so split long prompts into text items.

## Step 4: See which kernel each module runs

1. Rebuild once with `--detailed` (`llm_build --profiling-detailed`). It's the same engine plus kernel names.
2. Run `python3 scripts/profile_engine_layers.py <spec_base.engine> <plugin.so> out.json --prefill 4096`
   on the host. It uses TensorRT's Python bindings, which ship with JetPack.
   - It times one verify step (4 tokens) and one prefill step through TensorRT's IProfiler.
   - It groups GEMM time by tactic.
3. Map each GEMM to its module by input K and output N, read from the layer info's `Inputs` /
   `Outputs` dimensions. For Qwen3.8-27B:

   | Module | K × N |
   |---|---|
   | MLP gate/up | 5120 × 17408 |
   | MLP down | 17408 × 5120 |
   | GDN qkv | 5120 × 10240 |
   | GDN z | 5120 × 6144 |
   | GDN out / attn o | 6144 × 5120 |
   | attn q | 5120 × 12288 |
   | attn k / v | 5120 × 1024 |
   | lm_head | 5120 × 248320 |

4. Check that INT8 really runs as INT8:
   - Good: `cutlass3x_sm100_…gemm_s8_s8_s32…` or `__myl_Fc_…` with Int8 inputs.
   - Bad: a plain f16 kernel on a quantized layer, which means the Q/DQ didn't fuse.

Thor results for Qwen3.8-27B INT8 + MTP:
- **Verify step:** 119 ms, 91% of it GEMM.
  - Kernels are near the bandwidth limit (~250 GB/s).
  - TRT, vLLM Triton and cuBLASLt all take about 100 ms per step here.
  - **Decode can't be sped up by kernel choice. Only fewer bytes (INT4) or MTP help.**
- **4K prefill:** 1.83 s, of which 1.31 s is GEMM.
  - Every GEMM used CUTLASS 64×128 1sm.
  - A microbench shows ~0.73 s is possible at 4K.

## Step 5: Tune prefill kernels for <4K and ≥4K

**Why.** TensorRT picks one tactic per layer per optimization profile, timed at the profile's opt
shape. Edge-LLM builds two profiles:
- prefill, with opt = `maxInputLen / 2` (`cpp/builder/llmBuilder.cpp:1041` vanilla, `:1201` spec base);
- decode / verify.

So a 64K engine's prefill kernels are tuned for 32K prompts. Measured on Thor (64 layers, GEMM only):

| Prefill opt | 4K prompt | 32K prompt |
|---|---|---|
| 32768 (default) | 1.07 s | **7.93 s** |
| 4096 | **0.73 s** | 10.76 s |
| two prefill profiles | 0.73 s | 7.93 s |

The best tactic differs by length. At 4K, Myelin `Fc 256×256` / CUTLASS 128×128 win. At 32K,
CUTLASS 64×128 1sm wins; Myelin 256×256 there is 1.8× slower on MLP gate/up.

**5a. Measure first, per module** (minutes, no full build):

```bash
python3 scripts/make_int8_gemm_onnx.py gemm_onnx      # edit SHAPES for the model; run where onnx is installed
scripts/trt_opt_shape_bench.sh gemm_onnx optshape 4096 32768   # host trtexec; -> optshape/opt_shape.csv
```

- The bench rotates weight copies so weights come from DRAM, not cache. Single-weight tests read
  too fast.
- From the CSV, sum time × layer count per module for both opt values at both prompt lengths.
- Report the table above for this model, then pick:

**5b. Split prefill into a short and a long profile (verified on Thor).**

Apply `patches/split-prefill-profile-v0.11.1.patch` to the Edge-LLM v0.11.1 source and rebuild the
runtime into its own build dir, so the stock build stays usable:

```bash
cd TensorRT-Edge-LLM && git apply <skill-dir>/patches/split-prefill-profile-v0.11.1.patch
mkdir -p build-opt && cd build-opt && cmake .. <same flags as Step 2> && make -j$(nproc) llm_build llm_inference
EDGELLM_SHORT_PREFILL_TOKENS=4096 build-opt/examples/llm/llm_build ...   # rest as in Step 3
```

What the patch does (3 files, on top of v0.11.1):
- **Builder** (`setupLLMOptimizationProfiles`): with `EDGELLM_SHORT_PREFILL_TOKENS=N`, adds
  **profile 2** = prefill [1, N] tuned at opt N.
  - Profile 0 stays the full-range prefill (opt = max/2), and profile 1 stays decode.
  - Without the variable, the build is unchanged.
- **Runtime** (`TrtEngineExecutor::prepare`): when an engine has 3 profiles, prefill steps of ≤ N
  tokens run on profile 2. No decoder changes are needed.
- It also adds `EDGELLM_PREFILL_OPT_TOKENS=N`, which moves the single prefill profile's opt.
  **Don't use it alone:** at opt 4096 it made 32K prefill 25% slower, and that build also chose
  slower verify kernels (+17 ms/step).
- **The engine must be run with the patched `llm_inference`.** A stock runtime only uses profile 0
  for prefill, so it loses the short-prompt speedup.

Measured on Qwen3.8-27B INT8 + MTP=3, batch 1, 64K / 64K, Thor:

| | Stock engine | Split, N = 4096 |
|---|---|---|
| 4K prefill | 1.80 s | **1.19 s** |
| 32K prefill | 13.7 s | 13.8 s |
| Verify step 4K / 32K | 124.7 / 133.1 ms | 124.1 / 132.9 ms |
| Engine size / build time | 27.0 GB / 13.5 min | 27.0 GB / 18.7 min |

- Pick N from the user's prompt mix. 4096 is the measured split point.
- The short-profile kernels round differently, so greedy outputs for short prompts change.
  Re-check accuracy (5c).
- On another device (e.g. Orin), apply the same patch and build there with that target's
  `EMBEDDED_TARGET` — for this exact model, `orin-handoff.md` has the full step-by-step.

**5c. Verify.**
- Rebuild with `--detailed`.
- Rerun `profile_engine_layers.py` at 4K and 32K: the prefill GEMMs should show the new tactics.
- Rerun Step 3's benchmark, and compare gsm8k accuracy against the old engine (greedy).
- To profile the short profile: `profile_engine_layers.py ... --prefill 4096 --prefill-profile 2`.
  Compare two engines with `scripts/compare_engine_profiles.py`.
- Report the before / after prefill for both lengths.

Kernels TensorRT chooses between, for the report:
- **CUTLASS 3.x sm100 INT8** (s8×s8→s32, fp16 out). It varies by:
  - tile: 64×128, 128×128, 128×256, 256×128, 256×256;
  - 1sm / 2sm;
  - TMA epilogue;
  - stream-K;
  - cluster shape.
- **Myelin-generated `__myl_Fc` kernels**, chosen per shape. They don't appear in `--verbose`
  tactic tables but often win.
- **FP16 xmma / cuBLAS gemv** for unquantized layers (lm_head, small projections).
- **INT4 AWQ** runs through `Int4GroupwiseGemmPluginV2` (dequant + FP16 GEMM). TRT's tactic choice
  doesn't apply there. Its prefill is 3.5–4× slower than INT8. Profile it before blaming shapes.

To see every candidate's timing for one shape:
`trtexec --onnx=<gemm>.onnx --stronglyTyped --shapes=x:<M>x<K> ... --verbose`, then read
"Sorted table of all evaluated tactics".

## Reference (Qwen3.8-27B, Thor, batch 1, MTP=3)

| Engine | Prefill 4K / 32K | Decode gsm8k / 4K / 32K | Runtime peak | gsm8k-100 |
|---|---|---|---|---|
| INT8 SQ + MTP (64K / 64K) | 1.8 s / 13.7 s | 21.9 / 19.8 / 17.0 tok/s | 46.3 GB | 95% |
| INT4 AWQ + MTP (131072 / 131072) | 6.3 s / 57.1 s | 28.5 / 22.3 / 19.4 tok/s | 56.3 GB | 95% |
| vLLM INT8 + MTP (for comparison) | 4.8 s / 59.5 s | 22.0 / 18.4 / 13.4 tok/s | – | – |

Orin (~205 GB/s vs Thor's 273 GB/s): expect decode roughly 25% slower. Not yet measured;
the build steps for the Orin target are in `orin-handoff.md`.

## Finishing

Report:
- engine dir and files;
- the shapes;
- build time and build / runtime peak memory;
- prefill and decode tok/s and MTP accept length;
- accuracy against FP16;
- the per-module kernel table, and what the Step 5 tuning changed;
- for another device, the exact `cmake` / `llm_build` / `visual_build` / `llm_inference` commands.
