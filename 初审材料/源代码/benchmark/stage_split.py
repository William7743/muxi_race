import sys
sys.path.insert(0, "/root/moe_contest")
import torch
import remote_bench as rb

cand = sys.argv[1]
suf = sys.argv[2]
for case in (1, 2, 3):
    cfg, pt, tensors = rb.make_inputs(case, 20260901 + case)
    mod = rb.load_submission(cand, "moe_split", suf + f"_c{case}")
    out = torch.zeros((pt, cfg["hidden"]), device="cuda", dtype=torch.float16)
    x, gate, up, down, routed, gs, go, gpo, gidx = tensors
    if hasattr(mod, "_get_stage1"):
        s1 = mod._get_stage1(cfg["hidden"], cfg["intermediate"], cfg["experts"], pt, int(gidx.numel()))
        ul = mod._get_workspace(x, cfg["intermediate"])
        for _ in range(5):
            s1(x, gate, up, gs, gpo, gidx, ul)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            s1(x, gate, up, gs, gpo, gidx, ul)
        en.record(); torch.cuda.synchronize()
        t1 = st.elapsed_time(en) / 50
        wd = mod.T.float32 if routed.dtype == torch.float32 else mod.T.float16
        s2 = mod._get_stage2(cfg["hidden"], cfg["intermediate"], cfg["experts"], pt, int(routed.numel()), int(gidx.numel()), wd)
        for _ in range(5):
            s2(ul, down, routed, gs, go, gpo, gidx, out)
        torch.cuda.synchronize()
        st.record()
        for _ in range(50):
            s2(ul, down, routed, gs, go, gpo, gidx, out)
        en.record(); torch.cuda.synchronize()
        t2 = st.elapsed_time(en) / 50
        print(f"RESULT case{case}: stage1={t1:.3f} stage2={t2:.3f} total={t1+t2:.3f}", flush=True)
    else:
        ms = rb.measure(mod, tensors, out, warmup=10, iters=50)
        print(f"RESULT case{case}: total={ms:.3f} (no stage split)", flush=True)
    del tensors, out, mod
    torch.cuda.empty_cache()
