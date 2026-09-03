import argparse, sys
sys.path.insert(0, "/root/moe_contest")
import torch
import remote_bench as rb
from race_stress2 import random_group_sizes, reference, build_custom_inputs, check

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True)
ap.add_argument("--suffix", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--case", type=int, default=1)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

cfg = rb.CASES[a.case]
gen = torch.Generator(device="cpu")
gen.manual_seed(999000 + a.seed * 17)
gs2 = random_group_sizes(cfg, gen)
pt2, t2 = build_custom_inputs(cfg, gs2, 555000 + a.seed * 19 + a.case, "cuda")
ref = reference(t2, cfg, pt2, "cuda")
mod = rb.load_submission(a.candidate, "moe_cand1", a.suffix)
out = torch.zeros((pt2, cfg["hidden"]), device="cuda", dtype=torch.float16)
bad = 0
for r in range(a.reps):
    rb.invoke(mod, t2, out)
    torch.cuda.synchronize()
    n = check(out, ref, f"s{a.seed}.r{r}")
    if n:
        bad += 1
        break
print(f"RESULT seed={a.seed} bad={bad}", flush=True)
