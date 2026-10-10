#!/bin/bash
# Isolated opt-level sweep: plain (non-Triton) SmolVLA engines, strongly typed, 512 MB workspace.
set -u
B=~/bundles/smolvla-base-split
O=~/audit-20261010/optlevel
cd $O
for g in decode prefill vision; do
  case $g in decode) f=smolvlm_expert_decode.onnx; levels="2 3 4 5";; prefill) f=smolvlm_expert_prefill.onnx; levels="2 3 5";; vision) f=smolvlm_vision.onnx; levels="2 3 5";; esac
  for L in $levels; do
    e=$O/$g.L$L.engine
    [ -f $e ] || /usr/bin/trtexec --onnx=$B/$f --stronglyTyped --builderOptimizationLevel=$L \
        --memPoolSize=workspace:512M --saveEngine=$e --skipInference > $O/$g.L$L.build.log 2>&1
    echo "built $g L$L rc=$? $(grep -o "Engine generation completed in [0-9.]* seconds" $O/$g.L$L.build.log)"
  done
done
for rep in 1 2; do
for g in decode prefill vision; do
  for e in $O/$g.L*.engine; do
    /usr/bin/trtexec --loadEngine=$e --useCudaGraph --noDataTransfers --warmUp=2000 --duration=15 \
        > $e.run$rep.log 2>&1
    echo "$(basename $e) rep$rep $(grep -E "GPU Compute Time: min" $e.run$rep.log | sed "s/.*GPU Compute Time: //")"
  done
done
done
echo DONE
