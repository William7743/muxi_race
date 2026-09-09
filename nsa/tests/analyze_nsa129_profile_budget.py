"""Audit existing counters and calculate fixed-OJ score headroom; no GPU run."""
import hashlib
import json
from pathlib import Path
from reconcile_oj_calibration import checker_points, score

root = Path(__file__).resolve().parents[1]
dest = root/"results/2026-09-09"
meta = json.loads((dest/"peer_v343_profile_metadata.json").read_text(encoding="utf-8"))
source = root/"probes/probe_nsa129_oj109_s8_packed_prefetch.py"
assert meta["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
cuda = (dest/"peer_v343_case12_generated.cu").read_text(encoding="utf-8")
cuda = "\n".join(line.rstrip() for line in cuda.splitlines()).rstrip()+"\n"
assert meta["kernel_source_sha256"] == hashlib.sha256(cuda.encode()).hexdigest()
assert meta["correct"] and meta["seed"] == 0 and meta["mode"] == "public"
assert meta["case"]["case_id"] == 12 and meta["cold"]
assert meta["warmup"] == 10 and meta["capture_calls"] == 1
assert meta["cache_flush_bytes"] == 256000000

def counters(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for group in records.values():
        for item in group:
            assert not item["isError"] and item["name"] not in result
            result[item["name"]] = item["data"]
    assert len(result) == 7
    assert result["Total Instructions"] == result["Compute Instructions"]+result["Memory Instructions"]
    return result

candidate = counters(dest/"peer_v343_counters.json")
parent = counters(dest/"pair_profiles/profile_pair_nsa109/1_period0_dumped_result.json")
old_meta = json.loads((dest/"pair_profiles/profile_pair_nsa109_source.metadata.json").read_text(encoding="utf-8"))
assert old_meta["case"] == meta["case"]
for key in ("seed","mode","cold","warmup","capture_calls","cache_flush_bytes"):
    assert old_meta[key] == meta[key]
assert old_meta["source_sha256"] == hashlib.sha256((root/"probes/probe_nsa109_s2_scalar_probability.py").read_bytes()).hexdigest()
assert candidate == parent, "Revisit the unchanged-count conclusion"

observed = json.loads((dest/"oj109_final.json").read_text(encoding="utf-8"))
points = checker_points(next(r for r in observed if r["meta"]["id"] == 141658))
assert len(points) == 14
total = sum(p["score"] for p in points.values())
assert total == 1204
target = 88*14
budget = []
for cid,p in points.items():
    # Threshold at which the continuous formula reaches an integer score.
    thresholds = {}
    for gain in (1,2,3):
        new = p["score"]+gain
        max_time = p["baseline_us"]*(100/new-1)
        thresholds[str(gain)] = dict(target_point_score=new,max_kernel_us=max_time,
            minimum_time_reduction_fraction=1-max_time/p["oj_us"])
    budget.append(dict(case_id=cid,shape=p["config"],oj_us=p["oj_us"],score=p["score"],
        thresholds=thresholds,ideal_zero_time_total=(total-p["score"]+100)/14))

def modeled_total(factor):
    return sum(score(p["baseline_us"],p["oj_us"]*factor) for p in points.values())

low, high = 0., 1.
for _ in range(70):
    mid = (low+high)/2
    if modeled_total(mid) >= target:
        low = mid
    else:
        high = mid
assert modeled_total(low) >= target and modeled_total(high) < target
report = dict(status="PASS",profile_origin="Peer343 single-window counters, not this task's new run",
    metrics=candidate,counters_equal_prior109=True,
    counter_interpretation="Same recorded instruction counts and zero private accesses; does not identify latency bottleneck",
    actual_best_oj=86.,target_oj=88.,required_integer_point_gain=target-total,
    any_single_case_sufficient=any(p["ideal_zero_time_total"]>=88 for p in budget),
    equal_fraction_time_reduction_model=1-low,
    modeled_score_after_uniform_reduction=modeled_total(low)/14,cases=budget,
    limitations=["Counter captures are not simultaneous; metadata binding does not guarantee counter isolation",
        "Instruction counts are not bytes, stalls, latency or time shares",
        "The uniform-reduction model is a score requirement, not an achievable speedup prediction",
        "OJ anchor times have rounded precision and device differs from16GB slice",
        "Real88 must be verified on OJ; no new OJ result here"])
(dest/"nsa129_profile_score_budget.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="cases"}))
