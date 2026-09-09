"""Source-bound target screen audit, with nonqualified timing retained."""
import hashlib
import json
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
       138:"probe_nsa138_h2_bounded.py",139:"probe_nsa139_h2_prefetch.py",140:"probe_nsa140_h2_order.py"}
points=checker_points(json.loads((dest/"oj133_verified_details.json").read_text(encoding="utf-8"))[0])
rows=[json.loads(s) for s in (dest/"nsa138_140_screen.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
assert len(rows)==8
assert {(r["module"],r["case_id"],r["mode"],r["seed"]) for r in rows}=={
    (f"submissions/nsa{v}.py",13,m,571) for v in names for m in ("public","current")}
guard=json.loads((dest/"nsa138_140_guarded.json").read_text(encoding="utf-8"))
assert guard["exit_code"]==0 and guard["reason"] is None
norm={}
qualified=0
for row in rows:
    v=int(row["module"].split("nsa")[1].split(".")[0])
    assert row["source_sha256"]==hashlib.sha256((root/"probes"/names[v]).read_bytes()).hexdigest()
    assert row["correct"] and row["index_dtype"]=="torch.int32" and row["shared_output"]
    assert row["case"]==dict(case_id=13,**points[13]["config"])
    assert row["guard_source_sha256"]=="fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
    ts,gs=row["samples_us"]["cupti"],row["guard_samples"]["cupti"]
    assert len(ts)==len(gs)==4
    good=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
          if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03 and .92<=g["reference_ratio"]<=1.08]
    qualified+=len(good)
    norm[v,row["mode"]]=statistics.median(good) if len(good)>=2 else None
    path=dest/f"artifacts138_140/nsa{v}_case13"
    artifact=json.loads(path.with_suffix(".artifact.json").read_text(encoding="utf-8"))
    assert artifact["source_sha256"]==row["source_sha256"] and artifact["module"]==row["module"]
    code=path.with_suffix(".cu").read_text(encoding="utf-8")
    code="\n".join(l.rstrip() for l in code.splitlines()).rstrip()+"\n"
    assert hashlib.sha256(code.encode()).hexdigest()==row["kernel_source_sha256"]==artifact["cuda_sha256"]
ratios={str(v):{m:(norm[v,m]/norm[128,m] if norm[v,m] is not None and norm[128,m] is not None else None)
    for m in ("public","current")} for v in (138,139,140)}
report=dict(status="PASS",reference_assertions=8,qualified_samples=qualified,total_samples=32,
            target_ratios=ratios,goal88_achieved=False,
            limitation="Single target/seed on16GB slice; not full validation or OJ score.")
(dest/"nsa138_140_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
