"""Bind target tests, reversed recheck and captured compiler evidence to probes."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points

root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
130:"probe_nsa130_d128_bs32_block_bounds.py",131:"probe_nsa131_d128_bs16_block_bounds.py",
132:"probe_nsa132_d128_bounded_packed_v.py",133:"probe_nsa133_d128_packed_v_control.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/n).read_bytes()).hexdigest() for v,n in names.items()}
oj=json.loads((dest/"oj109_final.json").read_text(encoding="utf-8"))
cases={i:dict(case_id=i,**p["config"]) for i,p in checker_points(next(r for r in oj if r["meta"]["id"]==141658)).items()}
def load(name):
    return json.loads((dest/name).read_text(encoding="utf-8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def cuda_sha(path):
    text="\n".join(l.rstrip() for l in path.read_text(encoding="utf-8").splitlines()).rstrip()+"\n"
    return hashlib.sha256(text.encode()).hexdigest()

summaries=[]
all_rows=[]
for stem,versions,ids,seed,rounds,guard in (
    ("nsa130_131_screen",(128,130,131),(3,6),521,3,"nsa130_131_guarded"),
    ("nsa131_recheck",(128,131),(3,),523,5,"nsa131_recheck_guarded"),
    ("nsa132_133_screen",(128,132,133),(6,),529,3,"nsa132_133_guarded")):
    rows=[json.loads(l) for l in (dest/(stem+".jsonl")).read_text(encoding="utf-8").splitlines() if l.strip()]
    keys={(r["module"],r["case_id"],r["seed"],r["mode"]) for r in rows}
    expected={(f"submissions/nsa{v}.py",i,seed,m) for v in versions for i in ids for m in ("public","current")}
    assert keys==expected and len(rows)==len(expected)
    assert load(guard+".json")["exit_code"]==0 and load(guard+".json")["reason"] is None
    medians={}
    qualified=0
    for r in rows:
        assert r["correct"] and r["source_sha256"]==hashes[r["module"]]
        assert r["index_dtype"]=="torch.int32"
        assert all(r["case"][k]==val for k,val in cases[r["case_id"]].items())
        samples,guards=r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
        assert len(samples)==len(guards)==rounds
        values=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(samples,guards)
            if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
            and .92<=g["reference_ratio"]<=1.08]
        qualified+=len(values)
        assert len(values)>=2
        medians[r["module"],r["case_id"],r["mode"]]=statistics.median(values)
        all_rows.append(r)
    ratios={}
    for v in versions:
        if v==128: continue
        target=3 if v==131 else 6
        ratios[str(v)]={m:medians[f"submissions/nsa{v}.py",target,m]/medians["submissions/nsa128.py",target,m] for m in ("public","current")}
    summaries.append(dict(dataset=stem,reference_assertions=len(rows),qualified_samples=qualified,
                          total_samples=len(rows)*rounds,ratios=ratios))

artifacts=[]
for folder,targets in (("130_131",((128,6),(130,6),(128,3),(131,3))),
                       ("132_133",((128,6),(132,6),(133,6)))):
    for v,i in targets:
        prefix=f"nsa{v}_case{i}"
        artifact=load(f"artifacts{folder}/{prefix}.artifact.json")
        ir=load(f"ir{folder}/{prefix}.metadata.json")
        assert artifact["module"]==f"submissions/nsa{v}.py"
        assert artifact["case"]==cases[i]
        assert artifact["source_sha256"]==hashes[artifact["module"]]
        assert artifact["compile_flags"] is None
        assert artifact["cuda_sha256"]==cuda_sha(dest/f"artifacts{folder}/{prefix}.cu")
        assert ir["kernel_source_sha256"]==artifact["cuda_sha256"]
        assert ir["library_sha256"]==artifact["library_sha256"]
        assert ir["ir_sha256"]==sha(dest/f"ir{folder}/{prefix}.ll")
        matches=[r for r in all_rows if r["module"]==artifact["module"] and r["case_id"]==i]
        assert matches and all(r["kernel_source_sha256"]==artifact["cuda_sha256"] for r in matches)
        text=(dest/f"ir{folder}/{prefix}.ll").read_text(encoding="utf-8")
        private=[dict(line=k,text=l.strip()) for k,l in enumerate(text.splitlines(),1)
            if "addrspace(5)" in l or ("call " in l and "@llvm.memcpy.p" in l and (".p5." in l or ".p5(" in l))]
        assert private==ir["private_ir_lines"]
        artifacts.append(dict(version=v,case_id=i,library_sha256=ir["library_sha256"],
            private_ir_matches=len(private),
            private_alloca_lines=[l["text"] for l in private if "alloca " in l["text"]],
            device_text=ir["device_metadata"]["sections"][".text"]))
report=dict(status="PASS",datasets=summaries,total_reference_assertions=sum(x["reference_assertions"] for x in summaries),
    total_qualified_samples=sum(x["qualified_samples"] for x in summaries),artifacts=artifacts,
    recommendation="Retain NSA128. NSA130/132/133 regress; NSA131 small gain did not reproduce.",
    user_reported_best=dict(version=128,submission_id=141726,score=86.57),
    goal88_achieved=False,
    limitation="Target-only validation, not full suite. Embedded private IR is not measured runtime spill traffic; OJ128 score is user-reported.")
(dest/"nsa130_133_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
