"""Audit composition fingerprints, S8 screen and all extended reference records."""
import hashlib
import json
import statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
def read(name): return json.loads((dest/name).read_text(encoding="utf-8"))
def lines(name): return [json.loads(s) for s in (dest/name).read_text(encoding="utf-8").splitlines() if s.strip()]
def cuda_hash(path):
    code=path.read_text(encoding="utf-8")
    return hashlib.sha256(("\n".join(l.rstrip() for l in code.splitlines()).rstrip()+"\n").encode()).hexdigest()
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",142:"probe_nsa142_h2_s4_combination.py",
       143:"probe_nsa143_s8_modulo.py",144:"probe_nsa144_s8_predicate.py"}
hashes={f"submissions/nsa{v}.py":hashlib.sha256((root/"probes"/n).read_bytes()).hexdigest() for v,n in names.items()}
points=checker_points(read("oj133_verified_details.json")[0])
guard=read("nsa142_144_guarded.json")
assert guard["exit_code"]==0 and guard["reason"] is None
reports=[]
for filename,versions,ids,seed,rounds,parent in (
    ("nsa143_144_screen.jsonl",(142,143,144),(12,),599,3,142),
    ("nsa142_full.jsonl",(128,142),tuple(range(1,15)),601,2,128)):
    rows=lines(filename)
    expected={(f"submissions/nsa{v}.py",i,m,seed) for v in versions for i in ids for m in ("public","current")}
    assert len(rows)==len(expected) and {(r["module"],r["case_id"],r["mode"],r["seed"]) for r in rows}==expected
    normal={}
    qualified=0
    for r in rows:
        assert r["correct"] and r["source_sha256"]==hashes[r["module"]]
        assert r["index_dtype"]=="torch.int32" and r["shared_output"]
        assert r["case"]==dict(case_id=r["case_id"],**points[r["case_id"]]["config"])
        assert r["guard_source_sha256"]=="fa3dddff48a47eb179712df3f2171b15325263bdd1f24eb6a12f15ae06a910bb"
        ts,gs=r["samples_us"]["cupti"],r["guard_samples"]["cupti"]
        assert len(ts)==len(gs)==rounds
        good=[t/((g["before_us"]+g["after_us"])/2) for t,g in zip(ts,gs)
              if max(g["before_us"],g["after_us"])/min(g["before_us"],g["after_us"])-1<=.03 and .92<=g["reference_ratio"]<=1.08]
        qualified+=len(good)
        normal[r["module"],r["case_id"],r["mode"]]=statistics.median(good) if len(good)>=2 else None
        if parent==142:
            v=int(r["module"].split("nsa")[1].split(".")[0])
            prefix=f"artifacts143_144/nsa{v}_case12"
            artifact=read(prefix+".artifact.json")
            assert artifact["module"]==r["module"] and artifact["source_sha256"]==r["source_sha256"]
            assert artifact["cuda_sha256"]==r["kernel_source_sha256"]==cuda_hash(dest/(prefix+".cu"))
    targets=ids if parent==142 else (11,13)
    ratios={}
    for v in versions:
        if v==parent: continue
        ratios[str(v)]={}
        for i in targets:
            ratios[str(v)][str(i)]={}
            for m in ("public","current"):
                p,c=normal[f"submissions/nsa{parent}.py",i,m],normal[f"submissions/nsa{v}.py",i,m]
                ratios[str(v)][str(i)][m]=c/p if p is not None and c is not None else None
    reports.append(dict(dataset=filename,references=len(rows),qualified=qualified,total_samples=len(rows)*rounds,
                        ratios=ratios,unqualified_rows=[list(k) for k,x in normal.items() if x is None]))
full=lines("nsa142_full.jsonl")
changed=set()
for i in range(1,15):
    ds={}
    for v in (128,142):
        ds[v]=cuda_hash(dest/f"sources142/nsa{v}_case{i}.cu")
        assert all(r["kernel_source_sha256"]==ds[v] for r in full if r["module"]==f"submissions/nsa{v}.py" and r["case_id"]==i)
    if ds[128]!=ds[142]: changed.add(i)
    if i==11: assert ds[142]==cuda_hash(dest/"sources134/nsa134_case11.cu")
    if i==13: assert ds[142]==cuda_hash(dest/"sources138/nsa138_case13.cu")
    if i==12: assert ds[142]==read("artifacts143_144/nsa142_case12.artifact.json")["cuda_sha256"]
assert changed=={11,13}
counts={}
for filename,base,shapes,seeds in (
    ("nsa142_h2_stress.jsonl",dict(kv_heads=2,q_heads=32,dim=64,selected_blocks=1,block_size=16,causal=1),
     {(b,l,2,64,16) for b,l in ((1,64),(1,128),(1,256),(1,257),(1,512),(1,513),(2,256),(2,257))},(509,510)),
    ("nsa142_s4_stress.jsonl",dict(selected_blocks=4,causal=1),
     {(2,512,1,64,16),(1,512,2,64,16),(1,513,1,64,16),(1,496,1,64,16),
      (1,64,1,64,16),(4,1024,1,64,16),(1,512,1,64,32),(1,512,1,128,16)},(751,752,753))):
    rows=lines(filename)
    assert len(rows)==len(shapes)*len(seeds)*2
    observed=set()
    for r in rows:
        assert r["status"]=="PASS" and r["mismatch_count"]==0 and r["nonfinite"]==0
        assert r["source_sha256"]==hashes[r["module"]]
        assert all(r["case"][k]==v for k,v in base.items())
        assert r["case"]["q_heads"]==16*r["case"]["kv_heads"]
        shape=tuple(r["case"][k] for k in ("B","seq_len","kv_heads","dim","block_size"))
        observed.add((r["module"],shape,r["seed"]))
    assert observed=={(f"submissions/nsa{v}.py",s,seed) for v in (128,142) for s in shapes for seed in seeds}
    counts[filename]=len(rows)
report=dict(status="PASS",hashes=hashes,datasets=reports,stress_assertions=counts,
    changed_cases=sorted(changed),donor_generated_sources_match=True,
    actual_best_oj=dict(version=133,submission=141741,score=86.64),goal88_achieved=False,
    limitation="142 full validation;143/144 target screens only. Local16GB slice, no new OJ result or guaranteed score.")
(dest/"nsa142_144_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
