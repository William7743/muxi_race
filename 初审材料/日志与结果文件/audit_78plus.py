import sys, json, re, hashlib, time
sys.path.insert(0, '.')
import urllib.request, ssl, urllib.error

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com"

def req(path, token, body=None, headers=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + "/api/" + path, data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token, **(headers or {})},
        method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout, context=_CTX) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def solve_pow(ch, action):
    rd, diff, cid = ch["randomData"], ch["difficulty"], ch["id"]
    prefix = "0" * diff
    n = 0
    t0 = time.time()
    while True:
        h = hashlib.sha256((rd + str(n)).encode()).hexdigest()
        if h.startswith(prefix):
            print(f"PoW solved ({action}): diff={diff} {time.time()-t0:.1f}s", file=sys.stderr)
            return {"id": cid, "nonce": n, "response": h}
        n += 1

# 1) PoW 登录
st, ch = req("proofOfWork/issueChallenge", "", {"action": "login"})
assert st in (200, 201), ch
powp = solve_pow(ch, "login")
email, password = open(".xpuoj_credentials", encoding="utf-8").read().strip().split(":", 1)
st, d = req("auth/login", "", {"email": email, "password": password},
            headers={"X-Proof-Of-Work": json.dumps(powp)})
assert st in (200, 201), (st, d)
token = d["token"]
print("login OK", file=sys.stderr)

# 2) 拉全量提交列表（分段取）
all_rows = {}
take = 50
seen_ids = set()
offset_probe = 1
for take_count in (50, 100, 200):
    st, d = req("contest/play/querySubmissions", token, {
        "locale": "zh_CN", "contestId": 5, "problemOrder": 1,
        "submitter": "self", "takeCount": take_count})
    if st != 200:
        print("query fail", st, str(d)[:200]); break
    items = d.get("data") or d.get("items") or d.get("list") or []
    if isinstance(items, dict): items = items.get("list") or items.get("items") or []
    for s in items:
        meta = s.get("meta", s)
        sid = s.get("submissionId") or meta.get("id")
        ds = meta.get("displayScore")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            all_rows[sid] = (ds or 0, meta.get("status"))
    if len(items) >= 1:
        break

ge78 = [(ds, sid, st) for sid, (ds, st) in all_rows.items() if ds and ds >= 78]
ge78.sort(reverse=True)
print(f"总提交数(取回)={len(all_rows)}, >=78 分: {len(ge78)}")

# 3) 逐份拉代码审计
BAN_PATTERNS = [
    (r"import_source", "T.import_source(禁令10)"),
    (r"call_extern", "T.call_extern(禁令11)"),
    (r"data_ptr\s*\(", "data_ptr(禁令7/8 跨调用缓存)"),
    (r"\bid\s*\(\s*\w+\s*\)\s*(==|in|not)", "id()缓存(禁令7/8)"),
    (r"num_stages\s*=\s*([2-9]\d*)", "Pipelined ns>=2(禁令4 隐式pipeline)"),
    (r"cuda\.Stream|torch\.cuda\.stream", "CUDA流(禁令4 显式async)"),
    (r"cudaMemcpyAsync|mcta?lass", "外部异步拷贝/外部库(禁令7/12)"),
    (r"torch\.matmul|torch\.mm\b|torch\.bmm|einsum|F\.linear", "PyTorch计算(禁令2/6)"),
    (r"@\s", "隐式@运算(需人工复核)"),
    (r"mctlass|mcTlass", "mcTlass(禁令12)"),
]
for ds, sid, st in ge78:
    st2, det = req("submission/getSubmissionDetail", token, {"locale": "zh_CN", "submissionId": str(sid)})
    meta = det.get("meta", det)
    code = (det.get("content") or {}).get("code", "")
    if not code:
        c2 = det.get("content")
        if isinstance(c2, dict): code = c2.get("code", "")
    print(f"===== sid={sid} ds={meta.get('displayScore')} status={meta.get('status')} code_len={len(code)} =====")
    hits = []
    for pat, label in BAN_PATTERNS:
        for m in re.finditer(pat, code):
            line_no = code[:m.start()].count("\n") + 1
            line = code.splitlines()[line_no - 1].strip()[:100]
            hits.append((label, line_no, line))
    if hits:
        for label, ln, line in hits[:12]:
            print(f"  [{label}] L{ln}: {line}")
    else:
        print("  CLEAN ✓")
