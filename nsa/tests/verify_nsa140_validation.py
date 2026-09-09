"""Source-bound recheck, boundary and full-suite audit for NSA140."""
import hashlib
import json
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
def read(name): return json.loads((dest/name).read_text(encoding="utf-8"))
def lines(name): return [json.loads(s) for s in (dest/name).read_text(encoding="utf-8").splitlines() if s.strip()]
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",138:"probe_nsa138_h2_bounded.py",140:"probe_nsa140_h2_order.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/n).read_bytes()).hexdigest() for v,n in names.items()}
points=checker_points(read("oj133_verified_details.json")[0])
guard=read("nsa140_validation_guarded.json")
assert guard["exit_code"]==0 and guard["reason"] is None
reports=[]
for filename,versions,ids,seed,rounds in (
    ("nsa138_140_recheck.jsonl",(140,138,128),(13,),577,4),
    ("nsa140_full.jsonl",(128,140),tuple(range(1,15)),581,2)):
    rows=lines(filename)
    expected={(f"submissions/nsa{v}.py",i,m,seed) for v in versions for i in ids for m in ("public","current")}
    assert len(rows)==len(expected)
    assert {(r["module"],r["case_id"],r["mode"],r["seed"]) for r in rows}==expected
    normal={}
    qualified=0
    for row in rows:
        assert row["correct"] and row["source_sha256"]==hashes[row["module"]]
        assert row["index_dtype"]=="torch.int32" and row["shared_output"]
        assert row["case"]==dict(case_id=row["case_id"],**points[row["case_id"]]["config"])
        assert row["guard_source_sha256"]=="fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
        ts,gs=row["samples_us"]["cupti"],row["guard_samples"]["cupti"]
        assert len(ts)==len(gs)==rounds
        good=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
              if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03 and .92<=g["reference_ratio"]<=1.08]
        qualified+=len(good)
        normal[row["module"],row["case_id"],row["mode"]]=statistics.median(good) if len(good)>=2 else None
    ratios={}
    for v in versions:
        if v==128: continue
        ratios[str(v)]={}
        for m in ("public","current"):
            p,c=normal["submissions/nsa128.py",13,m],normal[f"submissions/nsa{v}.py",13,m]
            ratios[str(v)][m]=c/p if p is not None and c is not None else None
    reports.append(dict(dataset=filename,references=len(rows),qualified=qualified,total_samples=len(rows)*rounds,
                        target_ratios=ratios,unqualified_rows=[list(k) for k,x in normal.items() if x is None]))
full=lines("nsa140_full.jsonl")
changed=set()
for i in range(1,15):
    digests={}
    for v in (128,140):
        code=(dest/f"sources140/nsa{v}_case{i}.cu").read_text(encoding="utf-8")
        code="\n".join(l.rstrip() for l in code.splitlines()).rstrip()+"\n"
        digest=hashlib.sha256(code.encode()).hexdigest()
        assert all(r["kernel_source_sha256"]==digest for r in full if r["module"]==f"submissions/nsa{v}.py" and r["case_id"]==i)
        digests[v]=digest
        if i==13:
            assert digest==read(f"artifacts138_140/nsa{v}_case13.artifact.json")["cuda_sha256"]
    if digests[128]!=digests[140]: changed.add(i)
assert changed=={13}
stress=lines("nsa140_stress.jsonl")
shapes=((1,64),(1,128),(1,256),(1,257),(1,512),(1,513),(2,256),(2,257))
assert len(stress)==32
assert {(r["module"],r["case"]["B"],r["case"]["seq_len"],r["update"]) for r in stress}=={
    (f"submissions/nsa{v}.py",b,l,u) for v in (128,140) for b,l in shapes for u in (0,1)}
for row in stress:
    assert row["status"]=="PASS" and row["mismatch_count"]==0 and row["nonfinite"]==0
    assert row["source_sha256"]==hashes[row["module"]]
    assert row["seed"]==509+row["update"]
    assert all(row["case"][k]==v for k,v in dict(kv_heads=2,q_heads=32,dim=64,selected_blocks=1,block_size=16,causal=1).items())
result=dict(status="PASS",source_hashes=hashes,datasets=reports,stress_assertions=32,changed_cases=sorted(changed),
            actual_best_oj=dict(version=133,submission=141741,score=86.64),goal88_achieved=False,
            limitation="Local16GB/25% slice. No OJ result for140; aggregate PASS does not silently qualify rejected timing samples.")
(dest/"nsa140_validation_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
