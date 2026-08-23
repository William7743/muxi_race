#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查询 XPUOJ 最近提交结果（只读）。用法:
    python3 xpuoj_query.py                 # 最近 15 条
    python3 xpuoj_query.py --take 30       # 最近 30 条
    python3 xpuoj_query.py --detail 123720 # 查看某 submission 详情
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import XPUOJClient, load_credentials


class _Args:
    email = None
    password = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--take", type=int, default=15)
    p.add_argument("--detail", type=int, default=None)
    args = p.parse_args()

    email, password = load_credentials(_Args())
    client = XPUOJClient(email, password)
    print(f"已登录: {email}")

    if args.detail:
        data = client.get_submission_detail(args.detail)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    data = client.query_submissions(take_count=args.take)
    subs = data.get("submissions") if isinstance(data, dict) else data
    if not subs:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
        return
    for s in subs:
        sid = s.get("id") or s.get("submissionId")
        status = s.get("status")
        score = s.get("displayScore", s.get("score"))
        t = s.get("submitTime") or s.get("createdAt")
        size = s.get("answerSize")
        print(f"{sid}\t{status}\t{score}\t{t}\tsize={size}")


if __name__ == "__main__":
    main()
