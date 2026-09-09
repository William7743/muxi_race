"""CPU-only checks: dry-run plan and refusal without a confirmed resource handoff."""
import json
import subprocess
import sys
from pathlib import Path
driver=Path(__file__).with_name("run_nsa127_validation.py")
dry=subprocess.run([sys.executable,str(driver),"--dry-run"],capture_output=True,text=True)
assert dry.returncode==0, dry.stderr
plan=json.loads(dry.stdout)
assert plan["dry_run"] and len(plan["stages"])==3
assert [s[0] for s in plan["stages"]]==[
    "stress_nsa127_bounds.py","calibrate_oj_protocol.py","dump_sources.py"]
blocked=subprocess.run([sys.executable,str(driver)],capture_output=True,text=True)
assert blocked.returncode==2
assert "--handoff-confirmed required" in blocked.stderr
assert "Loading tilelang" not in blocked.stdout+blocked.stderr
print("PASS: dry-run plan; GPU workflow refused without handoff confirmation")
