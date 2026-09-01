import argparse, sys
sys.path.insert(0, "/root/moe_contest")
import torch
import remote_bench as rb
from race_stress2 import reference, check

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True)
ap.add_argument("--suffix", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--case", type=int, default=2)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

cfg = rb.CASES[a.case]
cfg, pt, t2 = rb.make_inputs(a.case, 777000 + a.seed * 23 + a.case)
ref = reference(t2, cfg, pt, "cuda")
mod = rb.load_submission(a.candidate, "moe_cand2", a.suffix)
out = torch.zeros((pt, cfg["hidden"]), device="cuda", dtype=torch.float16)
bad = 0
for r in range(a.reps):
    rb.invoke(mod, t2, out)
    torch.cuda.synchronize()
    n = check(out, ref, f"c{a.case}.s{a.seed}.r{r}")
    if n:
        bad += 1
        break
print(f"RESULT case={a.case} seed={a.seed} bad={bad}", flush=True)
