"""Verify source-bound controls, full-suite scope, stress evidence and score limits."""
import difflib, hashlib, json, math
from pathlib import Path
from reconcile_oj_calibration import checker_points, score

root = Path(__file__).resolve().parents[1]
dest = root/"results/2026-09-09"
files = {123:"probe_nsa123_d64_plain_block_bounds.py",124:"probe_nsa124_plain_bounds_all_grids.py",
         125:"probe_nsa125_d64_bounded_k_before_q.py",126:"probe_nsa126_d64_k_before_q_control.py",
         127:"probe_nsa127_combined_d64_bounds.py"}
hashes = {f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/f).read_bytes()).hexdigest() for v,f in files.items()}
def read(name):
    return json.loads((dest/name).read_text(encoding="utf-8"))
def rows(name):
    return [json.loads(x) for x in (dest/name).read_text(encoding="utf-8").splitlines() if x.strip()]
def qualifying(r):
    return sum(max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
        and .92<=g["reference_ratio"]<=1.08 for g in r["guard_samples"]["cupti"])
batches = {}; allrows = []
for name,cases,versions,seed,rounds in (
    ("nsa124",(2,4,14),(123,124),467,3),
    ("nsa125_126",(7,9),(123,126,125),479,3),
    ("nsa127_recheck",(4,5,7,9,14),(127,123),487,4),
    ("nsa127_full",tuple(range(1,15)),(123,127),491,2)):
    rr = rows(name+".jsonl")
    assert len(rr) == len(cases)*len(versions)*2
    assert {(r["case_id"],r["module"],r["mode"]) for r in rr} == {
        (c,f"submissions/nsa{v}.py",m) for c in cases for v in versions for m in ("public","current")}
    for r in rr:
        assert r["source_sha256"]==hashes[r["module"]] and r["correct"] and r["seed"]==seed
        assert len(r["samples_us"]["cupti"])==len(r["guard_samples"]["cupti"])==rounds
    batches[name] = dict(reference_checks=len(rr),qualified_samples=sum(map(qualifying,rr)),
                        total_samples=len(rr)*rounds)
    allrows += rr
full = rows("nsa127_full.jsonl")
changed = set()
for cid in range(1,15):
    for mode in ("public","current"):
        rr = [r for r in full if r["case_id"]==cid and r["mode"]==mode]
        if rr[0]["kernel_source_sha256"]!=rr[1]["kernel_source_sha256"]:
            changed.add(cid)
assert changed == {4,7,9,14}
for version in (123,127):
    for cid in (4,5,7,9,14):
        source = (dest/f"sources127/nsa{version}_case{cid}.cu").read_text(encoding="utf-8")
        source = "\n".join(x.rstrip() for x in source.splitlines()).rstrip()+"\n"
        digest = hashlib.sha256(source.encode()).hexdigest()
        assert all(r["kernel_source_sha256"]==digest for r in allrows
                   if r["module"]==f"submissions/nsa{version}.py" and r["case_id"]==cid)
        if version==127 and cid in (4,14,7,9):
            ancestor = 124 if cid in (4,14) else 125
            assert all(r["kernel_source_sha256"]==digest for r in allrows
                       if r["module"]==f"submissions/nsa{ancestor}.py" and r["case_id"]==cid)
stress = rows("nsa127_stress.jsonl")
assert len(stress)==72 and len({tuple(sorted(r["case"].items())) for r in stress})==12
for r in stress:
    assert r["source_sha256"]==hashes[r["module"]]
    assert r["status"]=="PASS" and r["mismatch_count"]==0 and r["nonfinite"]==0
oj = next(r for r in read("oj109_final.json") if r["meta"]["id"]==141658)
assert oj["meta"]["status"]=="Accepted"
points = checker_points(oj)
ratios = {cid:1.0 for cid in points}
ratios[5] = read("nsa123_recheck_summary.json")["geometric_relative_time"]
recheck = read("nsa127_recheck_summary.json")
for cid in changed:
    pp = [p for p in recheck["pairs"] if p["case_id"]==cid]
    assert len(pp)==2 and all(p["usable"] for p in pp)
    ratios[cid] = math.sqrt(pp[0]["candidate_relative_time"]*pp[1]["candidate_relative_time"])
projected = {cid:score(p["baseline_us"],p["oj_us"]*ratios[cid]) for cid,p in points.items()}
report = dict(status="PASS",hashes=hashes,batches=batches,stress_assertions=72,stress_shapes=12,
              changed_formal_cases_relative_to123=sorted(changed),
              fixed_parent_ratios=ratios,projected_points=projected,
              fixed_parent_projected_score=sum(projected.values())/14,
              actual_best_oj_score=86.0,
              limitation="Local16GB/25% sGPU evidence; projection is not a new OJ result. Goal88 not achieved.")
for v in (124,125,126,127):
    diff = "".join(difflib.unified_diff((root/"probes"/files[123]).read_text(encoding="utf-8").splitlines(True),
        (root/"probes"/files[v]).read_text(encoding="utf-8").splitlines(True),
        fromfile=files[123],tofile=files[v],n=0))
    (dest/f"nsa{v}_vs123.diff").write_text(diff,encoding="utf-8")
(dest/"nsa127_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
