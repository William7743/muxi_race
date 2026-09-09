"""Linux supervisor: yield only our new process group if competing work appears.

Use after observing prior peer workflows terminal. This is not a shared lock.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def outside_python(records, owned_pgid, supervisor_pid):
    # Compiler/runtime helpers may create a new session: PGID alone is not ownership.
    owned = {supervisor_pid, owned_pgid}
    owned.update(pid for pid, ppid, pgid, comm in records if pgid == owned_pgid)
    while True:
        descendants = {pid for pid, ppid, pgid, comm in records if ppid in owned}
        if descendants <= owned:
            break
        owned |= descendants
    return [pid for pid, ppid, pgid, comm in records if pid not in owned
            and comm.lower().startswith("python")]

def processes():
    result = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            data = path.read_text()
            end = data.rindex(")")
            fields = data[end+2:].split()
            if fields[0] != "Z":
                result.append((int(path.parent.name), int(fields[1]), int(fields[2]), data[data.index("(")+1:end]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer-terminal-observed", action="store_true")
    p.add_argument("--log", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--max-used-gib", type=float, default=28)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.peer_terminal_observed:
        p.error("Peer completion must be observed before launching")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or os.name != "posix":
        p.error("Linux and a child command required")
    for path in (Path(a.log), Path(a.metadata)):
        if path.exists():
            p.error(f"Refuse existing artifact: {path}")
    cg = Path("/sys/fs/cgroup/memory")
    limit = int((cg/"memory.limit_in_bytes").read_text())
    used = lambda: int((cg/"memory.usage_in_bytes").read_text())
    foreign = outside_python(processes(), -1, os.getpid())
    if foreign or limit-used() < 24*1024**3:
        p.error("Preflight failed: competing Python or insufficient host-memory headroom")
    events = []
    start = time.monotonic()
    env = dict(os.environ, NSA_VALIDATION_GUARD_PID=str(os.getpid()))
    with Path(a.log).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        reason = None
        events.append(dict(event="launch",child_pid=child.pid,used_bytes=used()))
        print(json.dumps(events[-1]), flush=True)
        try:
            while child.poll() is None:
                snapshot = processes()
                foreign = outside_python(snapshot, child.pid, os.getpid())
                memory = used()
                if foreign:
                    reason = "competing_python_detected"
                elif memory > min(a.max_used_gib*1024**3, limit-2*1024**3):
                    reason = "host_memory_high_water"
                elif time.monotonic()-start > a.timeout:
                    reason = "timeout"
                if reason:
                    events.append(dict(event="yield",reason=reason,foreign_pids=foreign,used_bytes=memory,
                        process_snapshot=[dict(pid=pid,ppid=ppid,pgid=pgid,comm=comm)
                            for pid,ppid,pgid,comm in snapshot]))
                    break
                time.sleep(.1)
        finally:
            if child.poll() is None:
                # Only the group created above is eligible for termination.
                assert os.getpgid(child.pid) == child.pid
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
        report = dict(exit_code=child.returncode,reason=reason,events=events,
                      elapsed_seconds=time.monotonic()-start,
                      limitation="Polling protection, not a peer lock or an OOM guarantee")
        Path(a.metadata).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report),flush=True)
    return 0 if child.returncode == 0 and reason is None else 1

if __name__ == "__main__":
    raise SystemExit(main())
