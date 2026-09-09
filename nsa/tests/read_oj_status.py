"""Bounded read-only OJ list query using existing local credentials; no submission."""
import argparse
import hashlib
import getpass
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xpuoj_submit import XPUOJClient, load_credentials
from xpuoj_feedback import summarize_submission


class Reader(XPUOJClient):
    def _login(self, email, password):
        # Domestic endpoint: avoid inherited proxy/PAC discovery and keep deadlines.
        self.session.trust_env = False
        print('phase=login_challenge', flush=True)
        proof = self._solve_proof_of_work('login')
        print('phase=login', flush=True)
        r = self.session.post(self.api + '/api/auth/login',
                              json={'email': email, 'password': password},
                              headers={'X-Proof-Of-Work': json.dumps(proof)}, timeout=(10, 15))
        if r.status_code not in (200, 201):
            raise RuntimeError('login HTTP ' + str(r.status_code))
        self.token = r.json().get('token')
        if not self.token:
            raise RuntimeError('login has no token; interactive verification may be required')
        self.session.headers['Authorization'] = 'Bearer ' + self.token

    def _solve_proof_of_work(self, action):
        r = self.session.post(self.api + '/api/proofOfWork/issueChallenge',
                              json={'action': action}, timeout=(10, 15))
        if r.status_code not in (200, 201):
            raise RuntimeError('challenge HTTP ' + str(r.status_code))
        c = r.json()
        difficulty = int(c['difficulty'])
        if not 0 <= difficulty <= 8:
            raise RuntimeError('challenge difficulty outside bounded client support')
        full, half = divmod(difficulty, 2)
        deadline = time.monotonic() + 20
        nonce = 0
        while time.monotonic() < deadline:
            digest = hashlib.sha256(f'{c["randomData"]}{nonce}'.encode()).digest()
            if not any(digest[:full]) and (not half or digest[full] < 16):
                return {'id': c['id'], 'nonce': nonce, 'response': digest.hex()}
            nonce += 1
        raise RuntimeError('normal login proof-of-work exceeded 20-second budget')


p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--email', help='Optional account email; password is read with hidden input')
p.add_argument('--details', nargs='+', type=int)
p.add_argument('--source-dir', help='Local private directory for downloaded submitted code')
p.add_argument('--raw-dir', help='Private .local directory for diagnostics; never publish raw responses')
a = p.parse_args()
try:
    if a.email:
        email, password = a.email, getpass.getpass('OJ password: ')
    else:
        email, password = load_credentials(SimpleNamespace(email=None, password=None))
    reader = Reader(email, password)
    del password
    if a.details:
        reports = []
        for sid in a.details:
            print('phase=detail id=' + str(sid), flush=True)
            response = reader.session.post(reader.api + '/api/submission/getSubmissionDetail',
                json={'locale': 'zh_CN', 'submissionId': str(sid)}, timeout=(10, 15))
            if response.status_code not in (200, 201):
                raise RuntimeError('detail HTTP ' + str(response.status_code))
            payload = response.json()
            if a.raw_dir:
                raw_folder = Path(a.raw_dir)
                if '.local' not in raw_folder.parts:
                    raise RuntimeError('raw responses require a private .local path')
                raw_folder.mkdir(parents=True, exist_ok=True)
                (raw_folder / ('detail_' + str(sid) + '.json')).write_text(json.dumps(payload), encoding='utf-8')
            safe = summarize_submission(payload, include_checker=True)
            progress = payload.get('progress', {})
            result_map = progress.get('testcaseResult', {})
            sample_diagnostics = []
            for sample in progress.get('samples', []):
                result = result_map.get(sample.get('testcaseHash'), {})
                item = {k: result[k] for k in ('status', 'time', 'checkerMessage')
                        if k in result and isinstance(result[k], (str, int, float))}
                error = result.get('userError', '')
                if isinstance(error, str) and error.startswith('Execution error: Language validation failed:'):
                    item['language_validation_error'] = error.splitlines()[0][:1000]
                sample_diagnostics.append(item)
            safe['sample_diagnostics'] = sample_diagnostics
            safe['requested_id'] = sid
            reports.append(safe)
            if a.source_dir:
                code = payload.get('content', {}).get('code')
                if isinstance(code, str):
                    folder = Path(a.source_dir)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / ('oj_' + str(sid) + '.py')).write_text(code, encoding='utf-8')
            Path(a.output).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
            print('detail_saved id=' + str(sid), flush=True)
        raise SystemExit(0)
    print('phase=query_submissions contest=7', flush=True)
    r = reader.session.post(reader.api + '/api/contest/play/querySubmissions',
                            json={'locale': 'zh_CN', 'contestId': 7, 'problemOrder': 1, 'takeCount': 50},
                            timeout=(10, 15))
    if r.status_code not in (200, 201):
        raise RuntimeError('query HTTP ' + str(r.status_code))
    data = r.json()
    rows = data.get('submissions') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError('unexpected list envelope keys=' + ','.join(data.keys()))
    keep = ('id', 'submissionId', 'status', 'displayScore', 'submitTime', 'createdAt', 'answerSize')
    safe = [{key: row[key] for key in keep if key in row and isinstance(row[key], (str,int,float,bool,type(None)))} for row in rows]
    report = {'queried_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'contest_id': 7, 'submissions': safe}
    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
except Exception as exc:
    # Avoid exposing response bodies, credential headers or session tokens.
    print('query_failed', type(exc).__name__, str(exc) if isinstance(exc, RuntimeError) else '', flush=True)
    raise SystemExit(1)
