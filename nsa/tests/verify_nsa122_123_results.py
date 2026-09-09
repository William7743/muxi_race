"""Bind measurements to immutable sources and check NSA123's exact changed case."""
import difflib
import hashlib
import json
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points, score

root = Path(__file__).resolve().parents[1]
dest = root / "results/2026-09-09"
files = {109:"probe_nsa109_s2_scalar_probability.py",
         122:"probe_nsa122_d64_block_bounds.py", 123:"probe_nsa123_d64_plain_block_bounds.py"}
hashes = {f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/f).read_bytes()).hexdigest()
          for v,f in files.items()}
def readrows(name):
    rows = [json.loads(line) for line in (dest/name).read_text().splitlines() if line.strip()]
    assert rows
    return rows
def checkrows(name, cases, versions, seed, rounds):
    rows = readrows(name)
    assert len(rows) == len(cases)*len(versions)*2
    assert {(r["case_id"],r["module"],r["mode"]) for r in rows} == {
        (c,f"submissions/nsa{v}.py",m) for c in cases for v in versions for m in ("public","current")}
    for r in rows:
        assert r["source_sha256"] == hashes[r["module"]] and r["correct"]
        assert r["seed"] == seed
        assert len(r["samples_us"]["cupti"]) == len(r["guard_samples"]["cupti"]) == rounds
    return rows
wide = checkrows("nsa122.jsonl",(5,7,8,9),(109,122),449,3)
recheck = checkrows("nsa123_recheck.jsonl",(5,),(109,123),457,4)
full = checkrows("nsa123_full.jsonl",range(1,15),(109,123),461,2)
changed = set()
for cid in range(1,15):
    for mode in ("public","current"):
        rr = [r for r in full if r["case_id"] == cid and r["mode"] == mode]
        assert len(rr) == 2
        if rr[0]["kernel_source_sha256"] != rr[1]["kernel_source_sha256"]:
            changed.add(cid)
assert changed == {5}, changed
exported = {}
for version in (109,122,123):
    source = (dest/f"sources123/nsa{version}_case5.cu").read_text()
    normalized = "\n".join(x.rstrip() for x in source.splitlines()).rstrip()+"\n"
    exported[version] = normalized
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    for row in wide+recheck+full:
        if row["module"] == f"submissions/nsa{version}.py" and row["case_id"] == 5:
            assert row["kernel_source_sha256"] == digest
assert exported[122] == exported[123]
assert "int64_t)" in exported[109] and "int64_t)" not in exported[123].replace("uint64_t)", "")
assert "block_id & 63" in exported[123] and "uint4 condval" not in exported[123]
for token in ("__syncwarp(", "__builtin_mxc_mma_16x16x16f16(", "tl::AllReduce<"):
    assert exported[109].count(token) == exported[123].count(token)
stress = readrows("nsa123_stress.jsonl")
assert len(stress) == 48
assert {r["module"] for r in stress} == {"submissions/nsa109.py","submissions/nsa123.py"}
assert len({tuple(sorted(r["case"].items())) for r in stress}) == 8
for r in stress:
    assert r["source_sha256"] == hashes[r["module"]]
    assert r["status"] == "PASS" and r["mismatch_count"] == 0 and r["nonfinite"] == 0
qualified = lambda r: sum(
    max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1 <= .03
    and .92 <= g["reference_ratio"] <= 1.08 for g in r["guard_samples"]["cupti"])
counts = {name:dict(reference_checks=len(rows), qualified_samples=sum(map(qualified,rows)),
                    total_samples=sum(len(r["samples_us"]["cupti"]) for r in rows))
          for name,rows in (("wide122",wide),("recheck123",recheck),("full123",full))}
oj = next(r for r in json.loads((dest/"oj109_final.json").read_text()) if r["meta"]["id"] == 141658)
assert oj["source"]["sha256"] == hashes["submissions/nsa109.py"] and oj["meta"]["status"] == "Accepted"
point = checker_points(oj)[5]
ratio = json.loads((dest/"nsa123_recheck_summary.json").read_text())["geometric_relative_time"]
projected = score(point["baseline_us"],point["oj_us"]*ratio)
report = dict(status="PASS", source_sha256=hashes, batches=counts,
              stress_assertions=len(stress), stress_shapes=8, changed_formal_cases=sorted(changed),
              nsa122_and_nsa123_case5_generated_source_identical=True,
              recheck_case5_relative_time=ratio,
              fixed_parent_projection=dict(case5_score=projected,total_score=86+(projected-point["score"])/14),
              limitation="Local C500 16GB/25%-compute slice; projection is not an OJ measurement or 88-point claim.")
for version in (122,123):
    diff = "".join(difflib.unified_diff((root/"probes"/files[109]).read_text(encoding="utf-8").splitlines(True),
        (root/"probes"/files[version]).read_text(encoding="utf-8").splitlines(True),
        fromfile=files[109],tofile=files[version],n=0))
    (dest/f"nsa{version}_vs109.diff").write_text(diff, encoding="utf-8")
(dest/"nsa123_verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
