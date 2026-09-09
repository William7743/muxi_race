"""Export generated sources for selected public shapes; run beside benchmark.py."""
import argparse
import json
from pathlib import Path
from benchmark import ROOT, load

p = argparse.ArgumentParser()
p.add_argument('--modules', nargs='+', required=True)
p.add_argument('--ids', nargs='+', type=int, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
dest = ROOT / a.output
dest.mkdir(parents=True, exist_ok=True)
cases = json.loads((ROOT / 'tests/oj_cases.json').read_text())
for module in a.modules:
    mod = load(ROOT / module)
    for case in cases:
        if case['case_id'] not in a.ids:
            continue
        args = [case[x] for x in ('B','seq_len','kv_heads','q_heads','dim','selected_blocks','block_size','causal')]
        kernel = mod._get_kernel(*args)
        (dest / f'{Path(module).stem}_case{case["case_id"]}.cu').write_text(kernel.get_kernel_source())
        print(module, case['case_id'], 'EXPORTED', flush=True)
