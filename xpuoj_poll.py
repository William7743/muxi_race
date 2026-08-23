#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轮询指定 submission 列表直到全部终态，结果打印为紧凑行。"""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import XPUOJClient, load_credentials

class A: email=None; password=None

TERMINAL = ("Accepted", "RuntimeError", "WrongAnswer", "TimeLimitExceeded",
            "MemoryLimitExceeded", "CompileError", "SystemError",
            "OutputLimitExceeded", "IdlenessLimitExceeded", "PresentationError",
            "Canceled", "InternalError", "JudgementFailed", "Failed")

ids = [int(x) for x in sys.argv[1:]]
email, password = load_credentials(A())
client = XPUOJClient(email, password)
pending = set(ids)
results = {}
deadline = time.time() + 3600
while pending and time.time() < deadline:
    for sid in sorted(pending):
        try:
            d = client.get_submission_detail(sid)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] {sid} query error: {e}", flush=True)
            continue
        meta = d.get("meta", {})
        status = meta.get("status")
        if status in TERMINAL:
            tr = d.get("progress", {}).get("testcaseResult", {})
            times = []
            for k, v in tr.items():
                ue = v.get("userError", "") or ""
                import re
                m = re.search(r'\{[^{}]*"time_ms"[^{}]*\}', ue)
                tc = v.get("input", "?")
                if m:
                    j = json.loads(m.group(0))
                    times.append(f"case{tc}:{j.get('tk_time_ms')}ms")
                else:
                    times.append(f"case{tc}:{v.get('status')}")
            results[sid] = (status, meta.get("displayScore"), times)
            print(f"[{time.strftime('%H:%M:%S')}] DONE {sid} {status} score={meta.get('displayScore')} {' '.join(times)}", flush=True)
            pending.discard(sid)
    if pending:
        time.sleep(120)
print("=== FINAL ===")
for sid in ids:
    if sid in results:
        s, sc, t = results[sid]
        print(sid, s, sc, " ".join(t))
    else:
        print(sid, "TIMEOUT")
