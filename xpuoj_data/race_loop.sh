#!/bin/bash
export PATH=/opt/conda/bin:$PATH PYTHONPATH=/opt/tilelang-metax-v0.1.10 MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib
cd /root/moe_contest
CAND=$1; SUF=$2; N=${3:-20}
> /root/moe_contest/loop_$SUF.log
for s in $(seq 0 $((N-1))); do
  timeout 300 python3 race_seed1.py --candidate $CAND --suffix ${SUF}_$s --seed $s >> /root/moe_contest/loop_$SUF.log 2>&1
  grep -E "RESULT|FAIL" /root/moe_contest/loop_$SUF.log | tail -2 >> /root/moe_contest/loop_$SUF.sum 2>/dev/null
  echo "done seed $s" >> /root/moe_contest/loop_$SUF.progress
done
echo "LOOP_DONE $CAND" >> /root/moe_contest/loop_$SUF.progress
