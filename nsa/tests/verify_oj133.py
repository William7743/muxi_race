"""Bind OJ141741 to NSA133 and distinguish changed-kernel from unchanged-path scores."""
import ast
import hashlib
import json
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points

root = Path(__file__).resolve().parents[1]
dest = root / "results/2026-09-09"
private = root / ".local"
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
captured = private / "oj141741_details.json"
report = read(captured if captured.exists() else dest / "oj133_verified_details.json")[0]
assert report["meta"]["id"] == 141741
assert report["meta"]["status"] == "Accepted"
assert report["meta"]["displayScore"] == 86.64 and report["missing_case_result_count"] == 0
names = {128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
         133:"probe_nsa133_d128_packed_v_control.py"}
probe = (root / "probes" / names[133]).read_text(encoding="utf-8")
download = private / "oj141741_source/oj_141741.py"
if download.exists():
    source = download.read_text(encoding="utf-8")
    binding = dict(leading_whitespace=source[:len(source)-len(source.lstrip())],
                   trailing_whitespace=source[len(source.rstrip()):])
else:
    binding = read(dest / "oj133_source_binding.json")
    assert all(not value.strip() for value in binding.values())
    source = binding["leading_whitespace"] + probe.strip() + binding["trailing_whitespace"]
assert source.strip() == probe.strip()
assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(probe))
assert hashlib.sha256(source.encode()).hexdigest() == report["source"]["sha256"]
new = checker_points(report)
old = checker_points(read(dest / "oj128_verified_details.json")[0])
rows = [json.loads(s) for s in (dest / "nsa133_full.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
assert len(rows) == 56
assert {(r["module"],r["case_id"],r["mode"],r["seed"]) for r in rows} == {
    (f"submissions/nsa{v}.py",i,m,563) for v in names for i in range(1,15) for m in ("public","current")}
guard = read(dest / "nsa133_full_guarded.json")
assert guard["exit_code"] == 0 and guard["reason"] is None
normal = {}
qualified = 0
fingerprints = {}
for r in rows:
    v = int(r["module"].split("nsa")[1].split(".")[0])
    assert r["source_sha256"] == hashlib.sha256((root/"probes"/names[v]).read_bytes()).hexdigest()
    assert r["correct"] and r["index_dtype"] == "torch.int32" and r["shared_output"]
    assert r["case"] == dict(case_id=r["case_id"], **new[r["case_id"]]["config"])
    assert r["guard_source_sha256"] == "fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
    ts,gs = r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
    assert len(ts) == len(gs) == 2
    good = [t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
            if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1 <= .03
            and .92 <= g["reference_ratio"] <= 1.08]
    qualified += len(good)
    # Retain failed qualification; do not promote one-sample timings or rerun
    # solely to remove an unfavorable observation.
    normal[v,r["case_id"],r["mode"]] = statistics.median(good) if len(good) == 2 else None
    key = v,r["case_id"]
    assert fingerprints.setdefault(key,r["kernel_source_sha256"]) == r["kernel_source_sha256"]
for (v,i),digest in fingerprints.items():
    text = (dest/f"sources133/nsa{v}_case{i}.cu").read_text(encoding="utf-8")
    normalized = "\n".join(l.rstrip() for l in text.splitlines()).rstrip()+"\n"
    assert hashlib.sha256(normalized.encode()).hexdigest() == digest
changed = {i for i in range(1,15) if fingerprints[128,i] != fingerprints[133,i]}
assert changed == {6}
deltas = []
for i in range(1,15):
    assert new[i]["config"] == old[i]["config"] and new[i]["baseline_us"] == old[i]["baseline_us"]
    deltas.append(dict(case_id=i,changed_kernel=i in changed,
        old_time_us=old[i]["oj_us"],new_time_us=new[i]["oj_us"],
        old_score=old[i]["score"],new_score=new[i]["score"],
        point_delta=new[i]["score"]-old[i]["score"],
        local_time_ratios={m:(normal[133,i,m]/normal[128,i,m]
            if normal[133,i,m] is not None and normal[128,i,m] is not None else None)
            for m in ("public","current")}))
total = sum(p["score"] for p in new.values())
assert total == 1213 and sum(p["point_delta"] for p in deltas if p["changed_kernel"]) == 0
assert [p["case_id"] for p in deltas if p["point_delta"]] == [13]
result = dict(status="PASS",submission_id=141741,version=133,score=86.64,
    source_equal_except_outer_whitespace=True,full_ast_equal=True,
    submitted_sha256=report["source"]["sha256"],
    probe_sha256=hashlib.sha256((root/"probes"/names[133]).read_bytes()).hexdigest(),
    full_reference_assertions=len(rows),qualified_samples=qualified,total_samples=112,
    unqualified_rows=[dict(version=v,case_id=i,mode=m) for (v,i,m),value in normal.items() if value is None],
    changed_cases=sorted(changed),case_deltas=deltas,
    total_integer_points=total,remaining_to88_integer_points=1232-total,
    changed_path_point_delta=0,unchanged_path_point_delta=1,goal88_achieved=False,
    conclusion="Verified best score, but all score gain occurs on unchanged case13; no demonstrated OJ benefit from packed-V change.",
    limitation="Rounded checker times and single OJ runs cannot establish causality. Local hardware is16GB/25% slice.")
for name,value in (("oj133_verified_details.json",[report]),("oj133_source_binding.json",binding),("oj133_verification.json",result)):
    (dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
