#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XPUOJ 提交脚本 — 沐曦 MoE 算子优化比赛 (contest 5, problem 1)
=============================================================
供 AI/用户通过命令行提交代码到 https://xpuoj.com/contest/5/problem/1

用法:
    python xpuoj_submit.py --code solution.py
    python xpuoj_submit.py --code solution.py --language tilelang.maca-c500
    python xpuoj_submit.py --code solution.py --contest 5 --problem 1
    python xpuoj_submit.py --code solution.py --status          # 提交后轮询评测结果

凭据:
    默认从环境变量 XPUOJ_EMAIL / XPUOJ_PASSWORD 读取，
    也可用 --email / --password 参数传入（不推荐，会留在 shell 历史）。
    也可以把凭据写入同目录 .xpuoj_credentials 文件（格式: email:password）。

返回:
    提交成功打印 submissionId；加 --status 会轮询到评测结束打印结果。
"""

import argparse
import hashlib
import json
import os
import sys
import time
import requests

API_HOST = os.environ.get(
    "XPUOJ_API",
    "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com",
)
DEFAULT_CONTEST = 5
DEFAULT_PROBLEM = 1
DEFAULT_LANGUAGE = "tilelang.maca-c500"


class XPUOJClient:
    """Minimal client for the XPUOJ Hydro API gateway."""

    def __init__(self, email, password, api_host=API_HOST):
        self.api = api_host
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Origin": "https://xpuoj.com",
            "Referer": "https://xpuoj.com/",
        })
        self.token = None
        self._login(email, password)

    def _login(self, email, password):
        proof_of_work = self._solve_proof_of_work("login")
        r = self.session.post(self.api + "/api/auth/login",
                              json={"email": email, "password": password},
                              headers={"X-Proof-Of-Work": json.dumps(proof_of_work)},
                              timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"登录失败 ({r.status_code}): {r.text[:300]}")
        data = r.json()
        self.token = data.get("token")
        if not self.token:
            raise RuntimeError(f"登录响应中没有 token: {data}")
        self.session.headers["Authorization"] = "Bearer " + self.token

    def _solve_proof_of_work(self, action):
        """Acquire and solve the same SHA-256 challenge used by the web client."""
        r = self.session.post(
            self.api + "/api/proofOfWork/issueChallenge",
            json={"action": action},
            timeout=30,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"获取登录校验失败 ({r.status_code}): {r.text[:300]}")
        challenge = r.json()
        difficulty = int(challenge["difficulty"])
        full_bytes, half_nibble = divmod(difficulty, 2)
        prefix = challenge["randomData"]
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{prefix}{nonce}".encode()).digest()
            if (not any(digest[:full_bytes]) and
                    (not half_nibble or digest[full_bytes] < 0x10)):
                return {
                    "id": challenge["id"],
                    "nonce": nonce,
                    "response": digest.hex(),
                }
            nonce += 1

    def post(self, path, body, proof_of_work_action=None, captcha_result=None):
        headers = {}
        if proof_of_work_action:
            proof = self._solve_proof_of_work(proof_of_work_action)
            headers["X-Proof-Of-Work"] = json.dumps(proof)
        if captcha_result:
            headers["X-Captcha-Result"] = json.dumps(captcha_result)
        r = self.session.post(
            self.api + "/api/" + path,
            json=body,
            headers=headers,
            timeout=60,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"API {path} 失败 ({r.status_code}): {r.text[:400]}")
        return r.json()

    # ---------- 业务接口 ----------

    def submit(self, code, contest_id=DEFAULT_CONTEST, problem_order=DEFAULT_PROBLEM,
               language=DEFAULT_LANGUAGE, compile_and_run_options=None):
        body = {
            "contestId": contest_id,
            "problemOrder": problem_order,
            "content": {
                "language": language,
                "code": code,
                "compileAndRunOptions": compile_and_run_options or {},
            },
        }
        captcha_token = os.environ.get("XPUOJ_TURNSTILE_TOKEN")
        captcha_result = ({"turnstile": {"token": captcha_token}}
                          if captcha_token else None)
        data = self.post(
            "contest/play/submit",
            body,
            "submit_problem",
            captcha_result,
        )
        return data

    def query_submissions(self, contest_id=DEFAULT_CONTEST, problem_order=DEFAULT_PROBLEM,
                          take_count=1):
        body = {
            "locale": "zh_CN",
            "contestId": contest_id,
            "problemOrder": problem_order,
            "takeCount": take_count,
        }
        return self.post("contest/play/querySubmissions", body)

    def get_submission_detail(self, submission_id):
        body = {"locale": "zh_CN", "submissionId": str(submission_id)}
        return self.post("submission/getSubmissionDetail", body)

    def wait_for_result(self, submission_id, timeout=600, interval=5):
        """轮询提交状态直到出现终态 (Accepted / RuntimeError / ...) 或超时。"""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            data = self.get_submission_detail(submission_id)
            meta = data.get("meta", data)
            status = meta.get("status")
            if status is None:
                status = data.get("status")
            if status and status != last:
                print(f"[{time.strftime('%H:%M:%S')}] status: {status}", flush=True)
                last = status
            if status in ("Accepted", "RuntimeError", "WrongAnswer", "TimeLimitExceeded",
                          "MemoryLimitExceeded", "CompileError", "SystemError",
                          "OutputLimitExceeded", "IdlenessLimitExceeded",
                          "PresentationError", "Canceled", "InternalError",
                          "JudgementFailed", "Failed"):
                return data
            time.sleep(interval)
        print(f"超时 ({timeout}s) 未等到终态", file=sys.stderr)
        return data


def load_credentials(args):
    email = args.email or os.environ.get("XPUOJ_EMAIL")
    password = args.password or os.environ.get("XPUOJ_PASSWORD")
    if not (email and password):
        cred_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 ".xpuoj_credentials")
        if os.path.exists(cred_file):
            with open(cred_file, "r", encoding="utf-8") as f:
                line = f.read().strip()
            if ":" in line:
                email, password = line.split(":", 1)
    if not (email and password):
        raise SystemExit(
            "缺少凭据: 请设置环境变量 XPUOJ_EMAIL / XPUOJ_PASSWORD, "
            "或用 --email/--password, 或创建 .xpuoj_credentials 文件 (格式 email:password)"
        )
    return email, password


def main():
    p = argparse.ArgumentParser(description="XPUOJ 代码提交工具")
    p.add_argument("--code", required=True, help="要提交的代码文件路径 (.py)")
    p.add_argument("--language", default=DEFAULT_LANGUAGE,
                   help=f"评测语言 (默认 {DEFAULT_LANGUAGE})")
    p.add_argument("--contest", type=int, default=DEFAULT_CONTEST, help="竞赛 ID")
    p.add_argument("--problem", type=int, default=DEFAULT_PROBLEM, help="题目序号")
    p.add_argument("--email", help="登录邮箱")
    p.add_argument("--password", help="登录密码")
    p.add_argument("--status", action="store_true", help="提交后轮询评测结果")
    p.add_argument("--timeout", type=int, default=600, help="等待评测的超时秒数")
    p.add_argument("--dry-run", action="store_true", help="只打印将发送的请求体, 不提交")
    args = p.parse_args()

    if not os.path.exists(args.code):
        raise SystemExit(f"代码文件不存在: {args.code}")
    with open(args.code, "r", encoding="utf-8") as f:
        code = f.read()

    if args.dry_run:
        print(json.dumps({
            "contestId": args.contest,
            "problemOrder": args.problem,
            "content": {"language": args.language, "code": code,
                        "compileAndRunOptions": {}},
        }, ensure_ascii=False, indent=2))
        return

    email, password = load_credentials(args)
    client = XPUOJClient(email, password)
    print(f"已登录: {email}")

    data = client.submit(code, args.contest, args.problem, args.language)
    print("提交响应:", json.dumps(data, ensure_ascii=False))

    # 从响应中找 submissionId
    submission_id = None
    if isinstance(data, dict):
        submission_id = (data.get("submissionId") or data.get("id")
                         or (data.get("submission") or {}).get("id"))
    if not submission_id:
        print("未能从响应中解析 submissionId, 响应内容:", data, file=sys.stderr)
        return

    print(f"提交成功: submissionId = {submission_id}")

    if args.status:
        result = client.wait_for_result(submission_id, timeout=args.timeout)
        print("\n=== 评测结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
