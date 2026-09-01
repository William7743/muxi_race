#!/bin/bash
export PATH=/opt/conda/bin:$PATH PYTHONPATH=/opt/tilelang-metax-v0.1.10 MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib
cd /root/moe_contest
CAND=$1; SUF=$2; CASE=$3; N=${4:-8}
> /root/moe_contest/loop_${SUF}_c$CASE.log
for s in $(seq 0 $((N-1))); do
  timeout 600 python3 race_seed2.py --candidate $CAND --suffix ${SUF}_${CASE}_$s --case $CASE --seed $s >> /root/moe_contest/loop_${SUF}_c$CASE.log 2>&1
  echo "done seed $s" >> /root/moe_contest/loop_${SUF}_c$CASE.progress
done
echo "LOOP_DONE case$CASE" >> /root/moe_contest/loop_${SUF}_c$CASE.progress
