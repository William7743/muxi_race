"""Check retained measurements without claiming completed candidate validation."""
import difflib, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
files={123:"probe_nsa123_d64_plain_block_bounds.py",124:"probe_nsa124_plain_bounds_all_grids.py",
       125:"probe_nsa125_d64_bounded_k_before_q.py",126:"probe_nsa126_d64_k_before_q_control.py",
       127:"probe_nsa127_combined_d64_bounds.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/f).read_bytes()).hexdigest() for v,f in files.items()}
def rows(name):
    return [json.loads(x) for x in (dest/name).read_text(encoding="utf-8").splitlines() if x.strip()]
checks=samples=0
for name,cases,versions,seed,rounds in (
    ("nsa124.jsonl",(2,4,14),(123,124),467,3),
    ("nsa125_126.jsonl",(7,9),(123,126,125),479,3),
    ("nsa127_recheck.jsonl",(4,5,7,9,14),(127,123),487,4)):
    rr=rows(name)
    assert len(rr)==len(cases)*len(versions)*2
    assert {(r["case_id"],r["module"],r["mode"]) for r in rr}=={
        (c,f"submissions/nsa{v}.py",m) for c in cases for v in versions for m in ("public","current")}
    for r in rr:
        assert r["correct"] and r["seed"]==seed and r["source_sha256"]==hashes[r["module"]]
        assert len(r["samples_us"]["cupti"])==len(r["guard_samples"]["cupti"])==rounds
        for g in r["guard_samples"]["cupti"]:
            assert max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03
            assert .92<=g["reference_ratio"]<=1.08
    checks+=len(rr);samples+=len(rr)*rounds
partial=rows("nsa127_stress_partial.jsonl")
assert len(partial)==60
assert len({tuple(sorted(r["case"].items())) for r in partial})==10
assert {(r["module"],r["update"]) for r in partial}=={
    (f"submissions/nsa{v}.py",u) for v in (123,127) for u in range(3)}
for r in partial:
    assert r["source_sha256"]==hashes[r["module"]]
    assert r["status"]=="PASS" and r["mismatch_count"]==r["nonfinite"]==0
assert "Killed" in (dest/"nsa127_validation_killed.log").read_text(encoding="utf-8")
assert "Killed" in (dest/"nsa127_validation_retry_killed.log").read_text(encoding="utf-8")
report=dict(measurement_integrity="PASS",candidate_validation="INCOMPLETE",
    paired_reference_assertions=checks,qualified_timing_samples=samples,
    retained_stress_assertions=len(partial),retained_stress_shapes=10,
    missing="Two stress shapes; complete 14-case regression; generated-source scope audit",
    actual_best_oj=86.0,goal88_achieved=False)
for v in (124,125,126,127):
    diff="".join(difflib.unified_diff((root/"probes"/files[123]).read_text(encoding="utf-8").splitlines(True),
        (root/"probes"/files[v]).read_text(encoding="utf-8").splitlines(True),
        fromfile=files[123],tofile=files[v],n=0))
    (dest/f"nsa{v}_vs123.diff").write_text(diff,encoding="utf-8")
(dest/"nsa127_partial_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
