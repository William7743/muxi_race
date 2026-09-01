import sys
sys.path.insert(0, "/root/moe_contest")
import torch
import remote_bench as rb

cand, suf, cases = sys.argv[1], sys.argv[2], sys.argv[3]
for case in [int(c) for c in cases.split(",")]:
    cfg, pt, tensors = rb.make_inputs(case, 20260901 + case)
    mod = rb.load_submission(cand, "moe_b23", suf + f"_c{case}")
    out = torch.zeros((pt, cfg["hidden"]), device="cuda", dtype=torch.float16)
    # correctness vs fp32 reference
    from race_stress2 import reference, check
    ref = reference(tensors, cfg, pt, "cuda")
    rb.invoke(mod, tensors, out)
    torch.cuda.synchronize()
    n = check(out, ref, f"c{case}")
    ms = rb.measure(mod, tensors, out, warmup=10, iters=100)
    print(f"RESULT {suf} case{case}: {ms:.3f} ms bad={n}", flush=True)
    del tensors, out, mod, ref
    torch.cuda.empty_cache()
