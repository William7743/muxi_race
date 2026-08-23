#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印指定 submission 的各用例状态与计时。用法: python3 xpuoj_case.py <id> [...]"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import XPUOJClient, load_credentials

class A: email=None; password=None

email, password = load_credentials(A())
client = XPUOJClient(email, password)
for sid in sys.argv[1:]:
    d = client.get_submission_detail(int(sid))
    meta = d.get("meta", {})
    tr = d.get("progress", {}).get("testcaseResult", {})
    print(f"== {sid} {meta.get('status')} score={meta.get('displayScore')}")
    for k, v in sorted(tr.items()):
        ue = v.get("userError", "") or ""
        m = re.search(r'\{[^{}]*"time_ms"[^{}]*\}', ue)
        if m:
            j = json.loads(m.group(0))
            print(f"  case{k}: {v.get('status')} tk={j.get('tk_time_ms')}ms "
                  f"tb={j.get('tb_time_ms')}ms pass={j.get('pass')}")
        else:
            print(f"  case{k}: {v.get('status')} {ue[:200]!r}")
