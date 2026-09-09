"""Verify preserved peer343 measurements bind to byte-exact NSA129.

Does not launch a GPU test or convert local measurements into actual OJ scores.
"""
import difflib
import hashlib
import json
import math
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points, score

root = Path(__file__).resolve().parents[1]
dest = root / "results/2026-09-09"
parent_path = root / "probes/probe_nsa109_s2_scalar_probability.py"
candidate_path = root / "probes/probe_nsa129_oj109_s8_packed_prefetch.py"
parent, candidate = "baselines/oj_141658.py", "submissions/nsa_v343.py"
hashes = {parent: hashlib.sha256(parent_path.read_bytes()).hexdigest(),
          candidate: hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
          "submissions/nsa_v176.py": "e1d8ba65bac2ddc601cfb05694a07fb562d97c88c456a9860914416938555981"}

def rows(name):
    return [json.loads(s) for s in (dest/name).read_text(encoding="utf-8").splitlines() if s.strip()]

def normalized(row):
    values, guards = row["samples_us"]["cupti"], row["guard_samples"]["cupti"]
    assert len(values) == len(guards)
    good = []
    for value, g in zip(values, guards):
        if (max(g["before_us"], g["after_us"])/min(g["before_us"], g["after_us"])-1 <= .03
                and .92 <= g["reference_ratio"] <= 1.08):
            good.append(value/((g["before_us"]+g["after_us"])/2))
    assert len(good) == len(values), "Recorded all-qualified claim no longer holds"
    return statistics.median(good)

full = rows("peer_v343_full_paired.jsonl")
confirm = rows("peer_v343_confirm_paired.jsonl")
assert len(full) == 42 and len(confirm) == 8
assert {(r["case_id"], r["module"], r["mode"], r["seed"]) for r in full} == {
    (i, m, "public", 0) for i in range(1,15) for m in hashes}
assert {(r["case_id"], r["module"], r["mode"], r["seed"]) for r in confirm} == {
    (12, m, mask, seed) for m in (parent,candidate) for mask in ("public","current") for seed in (42,137)}
for dataset, rounds in ((full,5),(confirm,7)):
    for r in dataset:
        assert r["correct"] and r["source_sha256"] == hashes[r["module"]]
        assert r["index_dtype"] == "torch.int32"
        assert r["guard_source_sha256"] == "93f4d85ffaf7f39425253ddb8fb87e2d8a3f49acac75350ec0c93e1cfba91d1c"
        assert len(r["samples_us"]["cupti"]) == rounds
        normalized(r)
changed = []
for cid in range(1,15):
    p = next(r for r in full if r["case_id"] == cid and r["module"] == parent)
    c = next(r for r in full if r["case_id"] == cid and r["module"] == candidate)
    if p["kernel_source_sha256"] != c["kernel_source_sha256"]:
        changed.append(cid)
assert changed == [12]
device_source = (dest/"peer_v343_case12_generated.cu").read_text(encoding="utf-8")
device_source = "\n".join(line.rstrip() for line in device_source.splitlines()).rstrip()+"\n"
case12 = next(r for r in full if r["case_id"] == 12 and r["module"] == candidate)
assert hashlib.sha256(device_source.encode()).hexdigest() == case12["kernel_source_sha256"]
assert "uint Vbits[8]" in device_source
assert "uint4 v_ = *(uint4*)(V +" in device_source
assert "*(uint4*)(Vbits + (r * 4))" in device_source
assert "make_uint2(Vbits[((r_1 * 4) + c)], Vbits[((r_1 * 4) + c)])" in device_source
assert "*(uint4*)(Ks_local_cast + 0)" in device_source
assert device_source.count("__syncwarp(") == 4
preload = device_source.index("uint4 v_ = *(uint4*)(V +")
qk = device_source.index("__builtin_mxc_mma_16x16x16f16")
unpack = device_source.index("make_uint2(Vbits[")
assert preload < qk < device_source.index("__syncwarp(",qk) < unpack
assert unpack < device_source.index("__syncwarp(",unpack) < device_source.index("half_t B_local_1[16]")
ratios = []
for seed in (42,137):
    for mask in ("public","current"):
        p = next(r for r in confirm if r["seed"] == seed and r["mode"] == mask and r["module"] == parent)
        c = next(r for r in confirm if r["seed"] == seed and r["mode"] == mask and r["module"] == candidate)
        ratios.append(normalized(c)/normalized(p))
ratio = math.prod(ratios)**(1/len(ratios))

current = rows("peer_v343_current_resume.jsonl")
values = rows("peer_v343_values.jsonl")
assert len(current) == 8 and len(values) == 12
assert {(r["case"]["seq_len"],r["pattern"]) for r in current} == {
    (length,pattern) for length in (1024,1040) for pattern in (
        "current_first","current_last_negative","current_then_previous","noncausal_current")}
assert {(r["module"],r["mode"],r["zero_q"],r["value_factor"]) for r in values} == {
    (m,mask,zero,factor) for m in (parent,candidate) for mask in ("public","current")
    for zero,factor in ((False,1),(False,16),(True,1024))}
for r in current + values:
    assert r["source_sha256"] == hashes[r["module"]]
    assert r["status"] == "PASS" and r["nonfinite"] == r["mismatch_count"] == 0
valid_current = [r for r in current if r["pattern"] == "current_first"]
assert len(valid_current) == 2
# Other patterns use negative/int64 indices or noncausal input, outside OJ contract.
# Boundary/extra-shape peer logs lacking source hashes are not certification here.
observed = json.loads((dest/"oj109_final.json").read_text(encoding="utf-8"))
anchor = checker_points(next(r for r in observed if r["meta"]["id"] == 141658))
for r in full + confirm:
    expected_shape = anchor[r["case_id"]]["config"]
    assert {k: r["case"][k] for k in expected_shape} == expected_shape
projected = {i: anchor[i]["score"] for i in anchor}
projected[12] = score(anchor[12]["baseline_us"], anchor[12]["oj_us"]*ratio)
report = dict(evidence_audit="PASS",origin="Peer workspace GPU runs; no new GPU execution by this task",
    candidate_source_sha256=hashes[candidate],parent_source_sha256=hashes[parent],
    case12_generated_source_sha256=case12["kernel_source_sha256"],
    generated_source_audit="Source-bound 128-bit V loads and shared writes; four original warp fences retained",
    source_changed_cases=changed,full_suite_reference_assertions=42,
    full_suite_candidate_assertions=14,full_suite_qualified_samples=210,
    confirmation_reference_assertions=8,confirmation_qualified_samples=56,
    source_bound_value_stress_assertions=12,current_mask_contract_assertions=2,
    excluded_out_of_contract_diagnostics=6,case12_relative_time=ratio,
    fixed_oj_projection=sum(projected.values())/14,projected_case12=projected[12],
    actual_best_oj=86.0,goal88_achieved=False,
    limitations=["16GB slice, not64GB OJ", "Full14-case paired run uses public mask and one seed",
        "No new OJ score", "Unchanged-source timing differences are not optimization gains",
        "No independently rerun GPU validation by this task"])
(dest/"nsa129_peer_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
(dest/"nsa129_vs109.diff").write_text("".join(difflib.unified_diff(
    parent_path.read_text(encoding="utf-8").splitlines(True),
    candidate_path.read_text(encoding="utf-8").splitlines(True),
    fromfile=parent_path.name,tofile=candidate_path.name,n=0)),encoding="utf-8")
print(json.dumps(report))
