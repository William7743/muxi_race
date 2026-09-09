"""Audit the D128 bound-only target screen; does not imply full validation."""
import difflib
import hashlib
import json
import statistics
from pathlib import Path

root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
files={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",130:"probe_nsa130_d128_bs32_block_bounds.py",131:"probe_nsa131_d128_bs16_block_bounds.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/f).read_bytes()).hexdigest() for v,f in files.items()}
rows=[json.loads(s) for s in (dest/"nsa130_131_screen.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
assert len(rows)==12
assert {(r["case_id"],r["module"],r["mode"],r["seed"]) for r in rows}=={
    (i,f"submissions/nsa{v}.py",m,521) for i in (3,6) for v in files for m in ("public","current")}
qualified=0
def norm(r):
    global qualified
    assert r["source_sha256"]==hashes[r["module"]] and r["correct"]
    assert r["index_dtype"]=="torch.int32"
    assert len(r["samples_us"]["cupti"])==len(r["guard_samples"]["cupti"])==3
    vv=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(r["samples_us"]["cupti"],r["guard_samples"]["cupti"])
        if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03 and .92<=g["reference_ratio"]<=1.08]
    qualified+=len(vv)
    return statistics.median(vv) if len(vv)>=2 else None
normalized={(r["case_id"],r["module"],r["mode"]):norm(r) for r in rows}
cuda={}
for v in files:
    for i in (3,6):
        source=(dest/f"sources130_131/nsa{v}_case{i}.cu").read_text(encoding="utf-8")
        source="\n".join(l.rstrip() for l in source.splitlines()).rstrip()+"\n"
        digest=hashlib.sha256(source.encode()).hexdigest()
        assert all(r["kernel_source_sha256"]==digest for r in rows if r["case_id"]==i and r["module"]==f"submissions/nsa{v}.py")
        cuda[v,i]=source
assert cuda[128,3]==cuda[130,3] and cuda[128,6]==cuda[131,6]
results=[]
for v,i in ((130,6),(131,3)):
    parent,candidate=cuda[128,i],cuda[v,i]
    assert parent!=candidate
    row=dict(version=v,case_id=i,ratios={})
    for mode in ("public","current"):
        p,c=normalized[i,"submissions/nsa128.py",mode],normalized[i,f"submissions/nsa{v}.py",mode]
        row["ratios"][mode]=c/p if p is not None and c is not None else None
    for token in ("__syncthreads(","__syncwarp(","__builtin_mxc_mma_16x16x16f16","exp2f("):
        assert parent.count(token)==candidate.count(token),(v,token)
    row["generated_counts"]={"parent_int64_tokens":parent.count("int64_t"),"candidate_int64_tokens":candidate.count("int64_t"),
        "parent_if_tokens":parent.count("if ("),"candidate_if_tokens":candidate.count("if (")}
    results.append(row)
    diff="".join(difflib.unified_diff((root/"probes"/files[128]).read_text(encoding="utf-8").splitlines(True),
        (root/"probes"/files[v]).read_text(encoding="utf-8").splitlines(True),fromfile=files[128],tofile=files[v],n=0))
    (dest/f"nsa{v}_vs128.diff").write_text(diff,encoding="utf-8")
report=dict(evidence_audit="PASS",reference_assertions=12,qualified_samples=qualified,total_samples=36,results=results,
    actual_best_oj=86.,goal88_achieved=False,limitation="Target screen only; no extended/full-suite validation or new OJ score")
(dest/"nsa130_131_screen_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
