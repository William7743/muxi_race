#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Caller supplies the same MACA/TileLang environment as the paired benchmark.
for version in nsa109 nsa110 nsa111; do
    dest="$PWD/results/profile_pair_${version}"
    if [[ -e "$dest" ]]; then
        echo "Refusing to overwrite prior profile: $dest" >&2
        exit 1
    fi
    /opt/mcProfiler-ubuntu18.04/mcProfiler perf_exec \
        --cmdline "python tests/profile_controlled_kernel.py --module submissions/${version}.py --case-id 12 --prefix results/profile_pair_${version}_source --seed 0 --mode public --cold" \
        --cwd "$PWD" --kernelname kernel --casename "pair_instructions_${version}" \
        --custom --counts 1 \
        --metrics "Total Instructions" "Compute Instructions" "Memory Instructions" \
                  "Global Read Instructions" "Global Write Instructions" \
                  "Private Read Instructions" "Private Write Instructions" \
        --output "$dest"
done
