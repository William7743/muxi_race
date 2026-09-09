"""Validate single-window profiling provenance; counters are not timing fractions."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/"results/2026-09-09"
def read(n): return json.loads((dest/n).read_text(encoding="utf-8"))
meta=read("profile128_d128_source.metadata.json")
assert meta["source_sha256"]==hashlib.sha256((root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_bytes()).hexdigest()
code=(dest/"profile128_d128_source.cu").read_text(encoding="utf-8")
code="\n".join(l.rstrip() for l in code.splitlines()).rstrip()+"\n"
digest=hashlib.sha256(code.encode()).hexdigest()
assert meta["kernel_source_sha256"]==digest==read("artifacts132_133/nsa128_case6.artifact.json")["cuda_sha256"]
assert meta["case"]==dict(case_id=6,B=8,seq_len=1024,kv_heads=1,q_heads=16,dim=128,selected_blocks=1,block_size=32,causal=1)
assert meta["correct"] and meta["seed"]==0 and meta["mode"]=="public"
assert meta["cold"] and meta["cache_flush_bytes"]==256000000
assert meta["warmup"]==10 and meta["capture_calls"]==1
guard=read("profile128_d128_guarded.json")
assert guard["exit_code"]==0 and guard["reason"] is None
metrics={}
for records in read("profile128_d128/1_period0_dumped_result.json").values():
    for item in records:
        assert not item["isError"] and item["name"] not in metrics
        metrics[item["name"]]=item["data"]
assert set(metrics)=={"ISU stall cycles layout","Private Read Instructions","Private Write Instructions",
                     "Global Memory Read bytes","Global Memory Write bytes","shared memory access efficiency","AP MMA Duty ratio"}
assert meta["expected_output_bytes"]==8*1024*16*128*2
for k,v in metrics.items():
    if isinstance(v,dict): assert all(x>=0 for x in v.values())
    else: assert v>=0
assert 0<=metrics["shared memory access efficiency"]<=100
assert 0<=metrics["AP MMA Duty ratio"]<=100
report=dict(status="PASS",metadata=meta,metrics=metrics,
    output_write_difference_bytes=metrics["Global Memory Write bytes"]-meta["expected_output_bytes"],
    limitations=[
        "PASS validates provenance and tool records, not exact counter accuracy or slice isolation.",
        "Write count exceeds expected tensor size by448 bytes; cause unverified, retained explicitly.",
        "AP MMA ratio is relative to AP active, not total kernel elapsed time.",
        "Shared non-conflict access proportion is not time spent or recoverable speedup.",
        "Stall categories are not proven disjoint or convertible to wall-clock fractions.",
        "Zero private instruction counters concern this capture only; not proof for all inputs.",
        "Profiler may replay its workload in multiple passes; metadata binds final replay.",
        "Only1_period0 counters used; process-wide report excluded."
    ],goal88_achieved=False)
(dest/"profile128_d128_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
