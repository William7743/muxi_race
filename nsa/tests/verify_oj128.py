"""Verify OJ141726 checker scores and source identity without executing source."""
import ast
import hashlib
import json
from pathlib import Path
from reconcile_oj_calibration import checker_points
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
private=root/".local"
captured=private/"oj128_verified_details.json"
report=json.loads((captured if captured.exists() else dest/"oj128_verified_details.json").read_text(encoding="utf-8"))[0]
assert report["meta"]["id"]==141726 and report["meta"]["status"]=="Accepted"
assert report["meta"]["displayScore"]==86.57 and report["missing_case_result_count"]==0
probe=(root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_text(encoding="utf-8")
download=private/"oj128_verified_source/oj_141726.py"
if download.exists():
    source=download.read_text(encoding="utf-8").replace("\r\n","\n").replace("\r","\n")
    stripped=source.strip()
    leading=source[:len(source)-len(source.lstrip())]
    trailing=source[len(source.rstrip()):]
    binding=dict(leading_whitespace=leading,trailing_whitespace=trailing,
        limitation="Reconstructs the fetched source using unchanged probe body and captured outer whitespace only")
else:
    binding=json.loads((dest/"oj128_source_binding.json").read_text(encoding="utf-8"))
    leading,trailing=binding["leading_whitespace"],binding["trailing_whitespace"]
    assert not leading.strip() and not trailing.strip()
    source=leading+probe.strip()+trailing
assert source.strip()==probe.strip()
assert ast.dump(ast.parse(source))==ast.dump(ast.parse(probe))
assert hashlib.sha256(source.encode()).hexdigest()==report["source"]["sha256"]
new=checker_points(report)
old=checker_points(next(r for r in json.loads((dest/"oj109_final.json").read_text(encoding="utf-8")) if r["meta"]["id"]==141658))
# Reconstruct changed paths from actual full-suite generated-source fingerprints.
fingerprints={}
names={109:"probe_nsa109_s2_scalar_probability.py",123:"probe_nsa123_d64_plain_block_bounds.py",
       127:"probe_nsa127_combined_d64_bounds.py",128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py"}
for name,versions in (("nsa123_full.jsonl",(109,123)),("nsa127_full.jsonl",(123,127)),("nsa128_full.jsonl",(127,128))):
    rows=[json.loads(l) for l in (dest/name).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows)==56
    assert {(r["module"],r["case_id"],r["mode"]) for r in rows}=={(f"submissions/nsa{v}.py",i,m) for v in versions for i in range(1,15) for m in ("public","current")}
    for r in rows:
        v=int(r["module"].split("nsa")[1].split(".")[0])
        assert r["correct"] and r["source_sha256"]==hashlib.sha256((root/"probes"/names[v]).read_bytes()).hexdigest()
        key=v,r["case_id"]
        assert fingerprints.setdefault(key,r["kernel_source_sha256"])==r["kernel_source_sha256"]
changed={i for i in range(1,15) if fingerprints[109,i]!=fingerprints[128,i]}
assert changed=={4,5,7,9,12,14}
deltas=[]
for i in range(1,15):
    assert new[i]["config"]==old[i]["config"] and new[i]["baseline_us"]==old[i]["baseline_us"]
    p=new[i]
    thresholds={str(p["score"]+j):p["baseline_us"]*(100/(p["score"]+j)-1)
                for j in (1,2,3) if p["score"]+j<=100}
    deltas.append(dict(case_id=i,old_time_us=old[i]["oj_us"],new_time_us=p["oj_us"],
        old_score=old[i]["score"],new_score=p["score"],score_delta=p["score"]-old[i]["score"],
        changed_kernel_path=i in changed,next_score_time_thresholds_us=thresholds))
total=sum(p["score"] for p in new.values())
assert total==1212 and 1232-total==20
summary=dict(status="PASS",submission_id=141726,verdict="Accepted",score=86.57,
    full_ast_equal=True,source_equal_except_outer_whitespace=True,
    submitted_lf_sha256=report["source"]["sha256"],
    probe_sha256=hashlib.sha256((root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_bytes()).hexdigest(),
    total_integer_points=total,target_integer_points=1232,remaining_integer_points=20,
    case_changes=deltas,changed_path_point_delta=sum(r["score_delta"] for r in deltas if r["changed_kernel_path"]),
    unchanged_path_point_delta=sum(r["score_delta"] for r in deltas if not r["changed_kernel_path"]),
    goal88_achieved=False,
    limitation="Checker times rounded to microseconds; causal attribution of score deltas is not established by two OJ submissions.")
# The reader already allowlists these fields; raw response/credentials never published.
(dest/"oj128_verified_details.json").write_text(json.dumps([report],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(dest/"oj128_source_binding.json").write_text(json.dumps(binding,indent=2)+"\n",encoding="utf-8")
(dest/"oj128_verification.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary))
