"""Full14-case source-bound audit of the conservative S4 synchronization candidate."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points, score
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
def read(n):
    return json.loads((dest/n).read_text(encoding="utf-8"))
rows=[json.loads(s) for s in (dest/"nsa134_full.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",134:"probe_nsa134_s4_manual_sync_conservative.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/n).read_bytes()).hexdigest() for v,n in names.items()}
points=checker_points(read("oj128_verified_details.json")[0])
assert len(rows)==56
assert {(r["case_id"],r["module"],r["mode"],r["seed"]) for r in rows}=={
    (i,f"submissions/nsa{v}.py",m,557) for i in range(1,15) for v in names for m in ("public","current")}
guard=read("nsa134_full_guarded.json")
assert guard["exit_code"]==0 and guard["reason"] is None
normal={}
qualified=0
changed=set()
for r in rows:
    assert r["correct"] and r["source_sha256"]==hashes[r["module"]]
    assert r["index_dtype"]=="torch.int32" and r["shared_output"]
    assert r["case"]==dict(case_id=r["case_id"],**points[r["case_id"]]["config"])
    ts,gs=r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
    assert len(ts)==len(gs)==2
    assert r["guard_source_sha256"]=="fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
    good=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
        if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
        and .92<=g["reference_ratio"]<=1.08]
    qualified+=len(good)
    normal[r["module"],r["case_id"],r["mode"]]=statistics.median(good) if len(good)>=2 else None
for i in range(1,15):
    fingerprints={}
    for v in names:
        text=(dest/f"sources134/nsa{v}_case{i}.cu").read_text(encoding="utf-8")
        text="\n".join(l.rstrip() for l in text.splitlines()).rstrip()+"\n"
        digest=hashlib.sha256(text.encode()).hexdigest()
        assert all(r["kernel_source_sha256"]==digest for r in rows if r["module"]==f"submissions/nsa{v}.py" and r["case_id"]==i)
        fingerprints[v]=digest
    if fingerprints[128]!=fingerprints[134]: changed.add(i)
assert changed=={11}
prior=read("nsa134_137_verification.json")
assert prior["status"]=="PASS" and prior["stress_assertions"]==48
ratios={}
for mode in ("public","current"):
    p,c=normal["submissions/nsa128.py",11,mode],normal["submissions/nsa134.py",11,mode]
    assert p is not None and c is not None
    ratios[mode]=c/p
projection={i:p["score"] for i,p in points.items()}
projection[11]=score(points[11]["baseline_us"],points[11]["oj_us"]*prior["multi_seed134_relative_time"])
report=dict(status="PASS",hashes=hashes,full_reference_assertions=56,
    full_qualified_samples=qualified,full_total_samples=112,changed_cases=[11],
    stress_assertions_from_target_audit=48,multi_seed_target_time_ratio=prior["multi_seed134_relative_time"],
    full_target_ratios=ratios,fixed_parent_projected_score=sum(projection.values())/14,
    actual_oj_score=86.57,actual_oj_submission=141726,goal88_achieved=False,
    limitation="Projection is not OJ feedback; four-way recheck differed from two-way tests. Shape/timing evidence is from16GB slice.")
(dest/"nsa134_full_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))

