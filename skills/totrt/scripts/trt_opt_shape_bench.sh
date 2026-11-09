#!/bin/bash
# Effect of the prefill profile's opt shape on INT8 GEMM time: for each shape build one engine with
# opt=<OPT> (min 1, max 32768) and time it at M=4096 and M=32768. Per-GEMM time = median / G.
#   trt_opt_shape_bench.sh <onnx dir> <work dir> <opt...>   -> <work dir>/opt_shape.csv
ONNX=$1; W=$2; shift 2; mkdir -p $W
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -f $W/opt_shape.csv ] || echo "shape,opt,M,G,us,tactic" > $W/opt_shape.csv
for OPT in "$@"; do
for f in $ONNX/*.onnx; do
  name=$(basename $f .onnx); G=$(cat $ONNX/$name.g)
  K=$(python3 -c "print({'gdn_out_proj':6144,'attn_o_proj':6144,'mlp_down':17408}.get('$name',5120))")
  eng=$W/${name}_opt$OPT.engine
  $TRTEXEC --onnx=$f --stronglyTyped --minShapes=x:1x$K --optShapes=x:${OPT}x$K --maxShapes=x:32768x$K \
    --profilingVerbosity=detailed --exportLayerInfo=$W/${name}_opt$OPT.layers.json --skipInference --saveEngine=$eng > $W/${name}_opt$OPT.build.log 2>&1 || { echo "$name opt=$OPT build failed"; continue; }
  tac=$(python3 -c "
import json,re;d=json.load(open('$W/${name}_opt$OPT.layers.json'))
t=[l.get('TacticName','') for l in d['Layers'] if 'Fc' in l.get('Name','') or 'gemm' in (l.get('TacticName') or '')]
print(re.sub(r'0x[0-9a-f]+_','',t[0])[:70] if t else '?')")
  for M in 4096 32768; do
    log=$W/${name}_opt${OPT}_m$M.log
    $TRTEXEC --loadEngine=$eng --shapes=x:${M}x$K --useCudaGraph --noDataTransfers --useSpinWait --warmUp=500 --duration=5 > $log 2>&1 || { echo "$name opt=$OPT M=$M failed"; continue; }
    med=$(grep -oP 'GPU Compute Time:.*median = \K[0-9.]+' $log)
    echo "$name,$OPT,$M,$G,$(python3 -c "print(f'{$med*1000/$G:.1f}')"),$tac" | tee -a $W/opt_shape.csv
  done
  rm -f $eng
done
done
