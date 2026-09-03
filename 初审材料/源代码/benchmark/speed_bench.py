import argparse, sys
sys.path.insert(0, "/root/moe_contest")
import torch
import remote_bench as rb

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True)
ap.add_argument("--suffix", required=True)
ap.add_argument("--cases", default="1,2,3")
a = ap.parse_args()

for case in [int(c) for c in a.cases.split(",")]:
    cfg, pt, tensors = rb.make_inputs(case, 20260901 + case)
    mod = rb.load_submission(a.candidate, "moe_bench", a.suffix + f"_c{case}")
    out = torch.zeros((pt, cfg["hidden"]), device="cuda", dtype=torch.float16)
    ms = rb.measure(mod, tensors, out, warmup=10, iters=100)
    print(f"RESULT case{case}: {ms:.3f} ms", flush=True)
    del tensors, out, mod
    torch.cuda.empty_cache()
