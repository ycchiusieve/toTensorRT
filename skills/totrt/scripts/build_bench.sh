#!/bin/bash
# Build TensorRT Edge-LLM engines from a `trtedge export` dir, then measure batch-1 latency and memory.
# Run on the device the engine is for (engines don't move between GPUs or TensorRT versions).
#
#   EDGELLM_BUILD=<TensorRT-Edge-LLM/build> build_bench.sh --onnx <export dir> --engine <engine dir> \
#     [--batch 1] [--input-len 65536] [--kv 65536] [--mtp 3] [--visual] [--detailed] [--bench in.json ...]
#
# --mtp N     spec_base + spec_draft (export needs --mtp); runs with N draft tokens, verify N+1
# --visual    also build visual/ (128..4096 image tokens per request, 1024 per image)
# --detailed  llm_build --profiling-detailed, so profile_engine_layers.py can name each kernel
set -o pipefail
BATCH=1; INLEN=65536; KV=65536; MTP=0; VISUAL=0; DETAILED=""; BENCH=()
while [ $# -gt 0 ]; do case $1 in
  --onnx) O=$2; shift 2;; --engine) E=$2; shift 2;; --batch) BATCH=$2; shift 2;;
  --input-len) INLEN=$2; shift 2;; --kv) KV=$2; shift 2;; --mtp) MTP=$2; shift 2;;
  --visual) VISUAL=1; shift;; --detailed) DETAILED=--profiling-detailed; shift;;
  --bench) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do BENCH+=("$1"); shift; done;;
  *) echo "unknown arg $1"; exit 2;; esac; done
: "${EDGELLM_BUILD:?set EDGELLM_BUILD to the TensorRT-Edge-LLM build dir}"; : "${O:?--onnx}"; : "${E:?--engine}"
export EDGELLM_PLUGIN_PATH=$EDGELLM_BUILD/libNvInfer_edgellm_plugin.so
BIN=$EDGELLM_BUILD/examples
mkdir -p $E/bench
pgrep -f "examples/(llm|multimodal)/(llm|visual)_build" >/dev/null && { echo "another engine build is running"; exit 1; }

memwatch() { ( while true; do free -m | awk 'NR==2{print $3}'; sleep 2; done ) > $1 & echo $!; }
build() {  # name log cmd...
  local name=$1 log=$2; shift 2; local s=$(date +%s) m=$(memwatch $E/$name.mem)
  "$@" > $log 2>&1; local rc=$?; kill $m
  echo "$name build: exit $rc, $(( $(date +%s)-s ))s, sys used peak $(sort -n $E/$name.mem | tail -1) MB (start $(head -1 $E/$name.mem) MB)"
  [ $rc -eq 0 ] || { tail -5 $log; exit 1; }
}

COMMON="--max-batch-size $BATCH --max-input-len $INLEN --max-kv-cache-capacity $KV $DETAILED"
if [ "$MTP" -gt 0 ]; then
  build base $E/llm_build.log $BIN/llm/llm_build --onnx-dir $O/llm --engine-dir $E $COMMON --max-verify-tree-size $((MTP+1)) --spec-base
  build draft $E/draft_build.log $BIN/llm/llm_build --onnx-dir $O/mtp_draft --engine-dir $E $COMMON --max-draft-tree-size $((MTP+1)) --spec-draft
  RUN="--spec-decode --spec-draft-top-k 1 --spec-draft-step $MTP --spec-verify-size $((MTP+1))"
else
  build llm $E/llm_build.log $BIN/llm/llm_build --onnx-dir $O/llm --engine-dir $E $COMMON
  RUN=""
fi
[ $VISUAL -eq 1 ] && build visual $E/visual_build.log $BIN/multimodal/visual_build --onnx-dir $O/visual --engine-dir $E/visual \
  --min-image-tokens 128 --max-image-tokens 4096 --max-image-tokens-per-image 1024
ls -la $E/*.engine $E/visual/*.engine 2>/dev/null

for t in "${BENCH[@]}"; do
  n=$(basename $t .json); m=$(memwatch $E/bench/$n.mem)
  $BIN/llm/llm_inference --engine-dir $E --input-file $t --output-file $E/bench/$n.out.json $RUN --warmup 1 --dump-profile > $E/bench/$n.log 2>&1
  rc=$?; kill $m
  echo "== $n: exit $rc, sys used peak $(sort -n $E/bench/$n.mem | tail -1) MB (start $(head -1 $E/bench/$n.mem) MB)"
  grep -E "Computed Tokens|Total Runs|Accept|Tokens/Second|Peak GPU" $E/bench/$n.log
done
