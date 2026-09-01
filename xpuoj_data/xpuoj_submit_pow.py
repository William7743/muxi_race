#!/usr/bin/env python3
"""OJ submitter with Proof-of-Work support."""
import hashlib, json, ssl, sys, time, urllib.request, urllib.error

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com"

def req(path, token, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + "/api/" + path, data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token, **(headers or {})},
        method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout, context=_CTX) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def login(email, password):
    st, d = req("auth/login", "", {"email": email, "password": password})
    return d["token"]

def solve_pow(token, action="submit_problem"):
    st, ch = req("proofOfWork/issueChallenge", token, {"action": action})
    if st not in (200, 201):
        raise RuntimeError(f"issueChallenge failed: {st} {ch}")
    rd, diff, cid = ch["randomData"], ch["difficulty"], ch["id"]
    t0 = time.time()
    prefix = "0" * diff
    n = 0
    while True:
        h = hashlib.sha256((rd + str(n)).encode()).hexdigest()
        if h.startswith(prefix):
            dt = time.time() - t0
            print(f"PoW solved: nonce={n} diff={diff} in {dt:.1f}s", file=sys.stderr)
            return {"id": cid, "nonce": n, "response": h}
        n += 1

def submit(token, code, contest_id=5, problem_order=1):
    powp = solve_pow(token, "submit_problem")
    hdr = {"X-Proof-Of-Work": json.dumps(powp)}
    body = {"contestId": contest_id, "problemOrder": problem_order,
            "content": {"language": "tilelang.maca-c500", "code": code, "compileAndRunOptions": {}}}
    st, d = req("contest/play/submit", token, body, headers=hdr)
    return st, d

def detail(token, sid):
    st, d = req("submission/getSubmissionDetail", token, {"submissionId": str(sid), "locale": "zh_CN"})
    return d

if __name__ == "__main__":
    token = open("/tmp/xpuoj_token.txt").read().strip()
    code = open(sys.argv[1]).read()
    st, d = submit(token, code)
    print("HTTP", st, "resp:", json.dumps(d))
