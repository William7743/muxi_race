"""Verify S4 synchronization screens, multi-seed recheck and compiled identity."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
def read(n):
    return json.loads((dest/n).read_text(encoding="utf-8"))
def lines(n):
    return [json.loads(s) for s in (dest/n).read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def cuda_sha(p):
    s="\n".join(l.rstrip() for l in p.read_text(encoding="utf-8").splitlines()).rstrip()+"\n"
    return hashlib.sha256(s.encode()).hexdigest()
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
134:"probe_nsa134_s4_manual_sync_conservative.py",135:"probe_nsa135_s4_manual_sync_reduce_fences.py",
136:"probe_nsa136_s4_manual_sync_safe_off.py",137:"probe_nsa137_s4_manual_sync_k_before_q.py"}
hashes={f"submissions/nsa{v}.py":sha(root/"probes"/n) for v,n in names.items()}
point=checker_points(read("oj128_verified_details.json")[0])[11]
expected_case=dict(case_id=11,**point["config"])
summaries=[]
all_rows=[]
for stem,versions,seeds,rounds,guard in (
    ("nsa134_135_screen",(128,134,135),(541,),3,"nsa134_135_guarded"),
    ("nsa134_137_screen",(128,134,136,137),(547,),4,"nsa134_137_guarded"),
    ("nsa134_multiseed",(128,134),(541,547,559),4,"nsa134_multiseed_retry_guarded")):
    rows=lines(stem+".jsonl")
    expected={(f"submissions/nsa{v}.py",s,m) for v in versions for s in seeds for m in ("public","current")}
    assert len(rows)==len(expected)
    assert {(r["module"],r["seed"],r["mode"]) for r in rows}==expected
    assert read(guard+".json")["exit_code"]==0 and read(guard+".json")["reason"] is None
    header=next(json.loads(l) for l in (dest/(guard+".log")).read_text(encoding="utf-8").splitlines() if l.startswith("{") and "reduce_header" in l)
    assert header["sha256"]=="ca5b49087fed84da9cd290ddf8d57578bce3f3bb20a89b61d0fd0739e3d5809f"
    normal={}
    qualified=0
    for r in rows:
        assert r["correct"] and r["case"]==expected_case and r["index_dtype"]=="torch.int32"
        assert r["source_sha256"]==hashes[r["module"]]
        assert r["guard_module"]=="submissions/nsa_v159.py"
        assert r["guard_source_sha256"]=="fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
        assert {x["guard_source_sha256"] for x in lines("nsa128_full.jsonl")}=={r["guard_source_sha256"]}
        ts,gs=r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
        assert len(ts)==len(gs)==rounds
        good=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
            if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
            and .92<=g["reference_ratio"]<=1.08]
        assert len(good)>=2
        qualified+=len(good)
        normal[r["module"],r["seed"],r["mode"]]=statistics.median(good)
    ratios={str(v):{f"{s}/{m}":normal[f"submissions/nsa{v}.py",s,m]/normal["submissions/nsa128.py",s,m]
                   for s in seeds for m in ("public","current")} for v in versions if v!=128}
    summaries.append(dict(dataset=stem,references=len(rows),qualified_samples=qualified,total_samples=len(rows)*rounds,ratios=ratios))
    all_rows+=rows
for folder,versions in (("134_135",(128,134,135)),("134_137",(128,134,136,137))):
    for v in versions:
        prefix=f"artifacts{folder}/nsa{v}_case11"
        m=read(prefix+".artifact.json")
        assert m["case"]==expected_case and m["source_sha256"]==hashes[m["module"]]
        assert m["compile_flags"] is None
        digest=cuda_sha(dest/(prefix+".cu"))
        assert digest==m["cuda_sha256"]
        assert all(r["kernel_source_sha256"]==digest for r in all_rows if r["module"]==f"submissions/nsa{v}.py")
ir={}
for v in (134,136):
    m=read(f"ir134_136/nsa{v}_case11.metadata.json")
    a=read(f"artifacts134_137/nsa{v}_case11.artifact.json")
    assert m["kernel_source_sha256"]==a["cuda_sha256"] and m["library_sha256"]==a["library_sha256"]
    assert m["ir_sha256"]==sha(dest/f"ir134_136/nsa{v}_case11.ll")
    ir[v]=m
assert ir[134]["device_metadata"]==ir[136]["device_metadata"]
assert ir[134]["fatbin_sha256"]==ir[136]["fatbin_sha256"]
stress=lines("nsa134_stress.jsonl")
shapes={(2,512,1,64,16),(1,512,2,64,16),(1,513,1,64,16),(1,496,1,64,16),
        (1,64,1,64,16),(4,1024,1,64,16),(1,512,1,64,32),(1,512,1,128,16)}
assert len(stress)==48
assert {(r["case"]["B"],r["case"]["seq_len"],r["case"]["kv_heads"],r["case"]["dim"],r["case"]["block_size"],r["module"],r["update"]) for r in stress}=={
    (*s,f"submissions/nsa{v}.py",u) for s in shapes for v in (128,134) for u in range(3)}
for r in stress:
    assert r["source_sha256"]==hashes[r["module"]] and r["status"]=="PASS"
    assert r["nonfinite"]==r["mismatch_count"]==0 and r["seed"]==751+r["update"]
    assert r["index_dtype"]=="torch.int32" and r["case"]["selected_blocks"]==4
    assert r["case"]["causal"]==1 and r["case"]["q_heads"]==16*r["case"]["kv_heads"]
ratio=math.prod(summaries[-1]["ratios"]["134"].values())**(1/6)
report=dict(status="PASS",datasets=summaries,stress_assertions=48,
    total_reference_assertions=len(all_rows)+48,total_qualified_samples=sum(x["qualified_samples"] for x in summaries),
    multi_seed134_relative_time=ratio,binary134_equals136=True,
    actual_best_oj=dict(version=128,submission_id=141726,score=86.57),
    goal88_achieved=False,full_suite_assessed_by_this_verifier=False,
    limitation="Timing qualification does not prove speedup; no new OJ result for134-137. Full formal suite is audited separately.")
(dest/"nsa134_137_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
