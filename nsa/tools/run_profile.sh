#!/usr/bin/env bash
# Run from the GPU host; credentials and remote connection setup stay external.
set -euo pipefail
NSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="${1:-baseline}"
case "$CANDIDATE" in
  baseline) KERNEL_NAME=kernel_kernel ;;
  gather) KERNEL_NAME=gather_kernel_kernel ;;
  *) echo 'Usage: bash nsa/tools/run_profile.sh baseline|gather' >&2; exit 2 ;;
esac
export MACA_PATH=/opt/maca
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/opt/tilelang-metax-v0.1.10/3rdparty/tvm/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH
PROFILE_DIR="$NSA_ROOT/results/profile-$(date +%Y%m%d-%H%M%S)-$CANDIDATE"
mkdir -- "$PROFILE_DIR"
cd -- "$PROFILE_DIR"
printf -v PROFILE_COMMAND '%q ' /opt/conda/bin/python -u "$NSA_ROOT/tools/profile_case.py" --candidate "$CANDIDATE" --capture
/opt/mcProfiler-ubuntu18.04/mcProfiler perf_exec \
  --cmdline "$PROFILE_COMMAND" --casename "nsa_s8_$CANDIDATE" \
  --cwd "$PROFILE_DIR" --kernelname "$KERNEL_NAME" \
  --per-kernel --custom --counts 1 --kernelnames "$KERNEL_NAME" \
  --metrics 'Total Cycles' 'Achieved waves' 'Global Memory Read bytes' \
  2>&1 | tee collection.log
