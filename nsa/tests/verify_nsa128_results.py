"""Source-bound combination audit; only completed datasets can pass."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points, score

root = Path(__file__).resolve().parents[1]
dest = root/"results/2026-09-09"
names = {127:"probe_nsa127_combined_d64_bounds.py",
         128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
         129:"probe_nsa129_oj109_s8_packed_prefetch.py"}
hashes = {f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/n).read_bytes()).hexdigest() for v,n in names.items()}

def rows(name):
    return [json.loads(s) for s in (dest/name).read_text(encoding="utf-8").splitlines() if s.strip()]

def normalized(r):
    samples,guards = r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
    assert len(samples)==len(guards)
    usable = [v/((g["before_us"]+g["after_us"])/2) for v,g in zip(samples,guards)
        if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
        and .92<=g["reference_ratio"]<=1.08]
    return (statistics.median(usable) if len(usable)>=2 else None),len(usable)

full,target,stress = [rows(n) for n in ("nsa128_full.jsonl","nsa128_s8.jsonl","nsa128_stress.jsonl")]
assert len(full)==56 and len(target)==6 and len(stress)==48
assert {(r["case_id"],r["module"],r["mode"],r["seed"]) for r in full} == {
    (i,f"submissions/nsa{v}.py",m,509) for i in range(1,15) for v in (127,128) for m in ("public","current")}
assert {(r["case_id"],r["module"],r["mode"],r["seed"]) for r in target} == {
    (12,f"submissions/nsa{v}.py",m,503) for v in (127,128,129) for m in ("public","current")}
oj = json.loads((dest/"oj109_final.json").read_text(encoding="utf-8"))
points = checker_points(next(r for r in oj if r["meta"]["id"]==141658))
for dataset,rounds in ((full,2),(target,3)):
    for r in dataset:
        assert r["source_sha256"]==hashes[r["module"]] and r["correct"]
        assert r["index_dtype"]=="torch.int32" and len(r["samples_us"]["cupti"])==rounds
        assert {k:r["case"][k] for k in points[r["case_id"]]["config"]} == points[r["case_id"]]["config"]
        normalized(r)
changed=set()
for i in range(1,15):
    for mode in ("public","current"):
        subset=[r for r in full if r["case_id"]==i and r["mode"]==mode]
        if subset[0]["kernel_source_sha256"] != subset[1]["kernel_source_sha256"]:
            changed.add(i)
assert changed=={12}
for mode in ("public","current"):
    equivalent=[r["kernel_source_sha256"] for r in target if r["mode"]==mode and r["module"] in (
        "submissions/nsa128.py","submissions/nsa129.py")]
    assert len(set(equivalent))==1
for v in (127,128):
    for i in range(1,15):
        source=(dest/f"sources128/nsa{v}_case{i}.cu").read_text(encoding="utf-8")
        source="\n".join(line.rstrip() for line in source.splitlines()).rstrip()+"\n"
        digest=hashlib.sha256(source.encode()).hexdigest()
        matching=[r for r in full+target if r["module"]==f"submissions/nsa{v}.py" and r["case_id"]==i]
        assert matching and all(r["kernel_source_sha256"]==digest for r in matching)
shapes={(1,1024,1,64,16),(4,1024,1,64,16),(1,1040,1,64,16),(1,1025,1,64,16),
        (1,1008,1,64,16),(1,1024,2,64,16),(1,1024,1,64,32),(1,1024,1,128,16)}
assert {(r["case"]["B"],r["case"]["seq_len"],r["case"]["kv_heads"],r["case"]["dim"],r["case"]["block_size"],
         r["module"],r["update"]) for r in stress} == {
    (*shape,f"submissions/nsa{v}.py",u) for shape in shapes for v in (127,128) for u in range(3)}
for r in stress:
    assert r["source_sha256"]==hashes[r["module"]] and r["status"]=="PASS"
    assert r["nonfinite"]==r["mismatch_count"]==0 and r["seed"]==701+r["update"]
    assert r["index_dtype"]=="torch.int32"
    c=r["case"]
    assert c["q_heads"]==c["kv_heads"]*16 and c["selected_blocks"]==8 and c["causal"]==1
ratios=[]
for mode in ("public","current"):
    p=next(r for r in target if r["mode"]==mode and r["module"]=="submissions/nsa127.py")
    c=next(r for r in target if r["mode"]==mode and r["module"]=="submissions/nsa128.py")
    pv,cv=normalized(p)[0],normalized(c)[0]
    if pv is not None and cv is not None:
        ratios.append(cv/pv)
assert len(ratios)==2,"No complete paired S8 comparison; do not project"
ratio=math.sqrt(math.prod(ratios))
parent=json.loads((dest/"nsa127_verification.json").read_text(encoding="utf-8"))
assert parent["status"]=="PASS"
projection=dict(parent["projected_points"])
projection["12"]=score(points[12]["baseline_us"],points[12]["oj_us"]*ratio)
report=dict(status="PASS",hashes=hashes,full_reference_assertions=56,full_qualified_samples=sum(normalized(r)[1] for r in full),
    full_total_samples=112,target_reference_assertions=6,target_qualified_samples=sum(normalized(r)[1] for r in target),
    target_total_samples=18,stress_assertions=48,stress_shapes=8,changed_cases_relative_to127=sorted(changed),
    s8_generated_source_matches129=True,s8_relative_time=ratio,fixed_parent_projected_score=sum(projection.values())/14,
    projected_points=projection,actual_best_oj=86.,goal88_achieved=False,
    limitation="16GB slice tests; projection is not a new OJ result. Full and target timing qualification may differ.")
(dest/"nsa128_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
