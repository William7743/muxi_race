"""Match downloaded submitted source without executing candidate code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repository', type=Path, required=True)
p.add_argument('--downloaded', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

def fingerprints(path):
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    sha = hashlib.sha256(text.encode()).hexdigest()
    tree = ast.parse(text)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
    return sha, hashlib.sha256(ast.dump(tree).encode()).hexdigest()

index = []
for folder in ('submissions', 'baselines', 'experiments'):
    for path in (a.repository / folder).glob('*.py'):
        try:
            sha, ast_sha = fingerprints(path)
        except (SyntaxError, UnicodeError):
            continue
        index.append((str(path.relative_to(a.repository)), sha, ast_sha))
report = []
for path in sorted(a.downloaded.glob('oj_*.py')):
    sha, ast_sha = fingerprints(path)
    row = dict(submission_id=int(path.stem.split('_')[1]), lf_sha256=sha,
               exact_matches=[name for name, h, _ in index if h == sha],
               executable_ast_matches=[name for name, _, h in index if h == ast_sha])
    report.append(row)
a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
