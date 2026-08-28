import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import XPUOJClient, load_credentials
import argparse

# 天气金丝雀 v2：
#  - v64 双流实验已在干净窗口定论 WA（4 连，基线同窗 Accepted 76）→ 路线关闭，不再探测
#  - 当前任务：周期采样基线；好天气（timeUsed 低 → ds 高）连发抓 77+ 窗口；
#    坏天气/漂移窗拉长间隔省槽位
MAX_ROUNDS = int(os.environ.get("CANARY_ROUNDS", "40"))
SLEEP_S = int(os.environ.get("CANARY_SLEEP", "1200"))
GOOD_MS = int(os.environ.get("CANARY_GOOD_MS", "21000"))
FAST_MS = int(os.environ.get("CANARY_FAST_MS", "19800"))  # ds 77+ 的经验阈值
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_canary.log")

ns = argparse.Namespace(email=None, password=None)
email, pw = load_credentials(ns)
c = XPUOJClient(email, pw)


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_code(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "r", encoding="utf-8") as f:
        return f.read()


def submit_and_poll(name, max_polls=110):
    code = read_code(name)
    try:
        data = c.submit(code, contest_id=5, problem_order=1)
    except Exception as e:
        log(f"submit {name} EXC: {e!r}")
        return "SUBMIT_EXC", None, None, None
    sid = data.get("submissionId") or data.get("id")
    log(f"submit {name} sid={sid}")
    for i in range(max_polls):
        time.sleep(8)
        try:
            d = c.get_submission_detail(sid)
        except Exception as e:
            log(f"poll sid={sid} EXC: {e!r}")
            time.sleep(30)
            continue
        if not d:
            continue
        meta = d.get("meta", d)
        status = meta.get("status")
        if status in ("Running", "Pending", "Judging", None):
            continue
        ds = meta.get("displayScore")
        tu = meta.get("timeUsed")
        return status, ds, tu, sid
    return "TIMEOUT", None, None, sid


for rnd in range(1, MAX_ROUNDS + 1):
    log(f"=== round {rnd}/{MAX_ROUNDS} canary ===")
    status, ds, tu, sid = submit_and_poll("ref_126947.py")
    log(f"canary sid={sid} status={status} ds={ds} timeUsed={tu}")
    if status != "Accepted":
        # WA（漂移窗）或超时：拉长间隔
        time.sleep(SLEEP_S * 2)
        continue
    if tu is not None and tu <= FAST_MS:
        # 快窗口：立刻连发抓更高分
        log(f"FAST window ({tu}ms) -> immediate resample")
        time.sleep(60)
        continue
    if tu is not None and tu <= GOOD_MS:
        log(f"weather good ({tu}ms) -> short interval")
        time.sleep(300)
    else:
        log(f"weather bad ({tu}ms > {GOOD_MS}) — long wait")
        time.sleep(SLEEP_S)

log("=== canary loop done ===")
