#!/bin/bash
# remote_moe_bench.sh — 在服务器上运行的 MoE 本地评测脚本
# 用法: bash remote_moe_bench.sh [case] [warmup] [iters]
cd /root/moe_contest

cat > /tmp/env_moe.sh << 'ENVEOF'
#!/bin/bash
export PATH=/opt/conda/bin:/opt/anaconda3/bin:$PATH
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVEOF

cp /root/moe_contest/env_moe.sh /root/env_moe.sh 2>/dev/null || true

/usr/bin/nohup bash -c '
. /root/env_moe.sh
export PATH="/opt/conda/bin:/opt/anaconda3/bin:\$PATH"
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/usr/local/lib:\$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -c "
import torch, tilelang
print(\"imports ok\")
torch.cuda.set_device(0)
print(\"device:\", torch.cuda.get_device_name(0))
"
' > /root/moe_contest/env_check.log 2>&1 &