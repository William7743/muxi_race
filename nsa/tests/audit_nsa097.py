"""Bounded audit of the loader-compatible rewrite; not the actual OJ sandbox."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--baseline', required=True)
p.add_argument('--candidate', required=True)
a = p.parse_args()
old_source = Path(a.baseline).read_text(encoding='utf-8')
new_source = Path(a.candidate).read_text(encoding='utf-8')
old, new = ast.parse(old_source), ast.parse(new_source)
removed = []
for node in old.body:
    if isinstance(node, ast.Assign):
        try:
            ast.literal_eval(node.value)
        except (ValueError, TypeError):
            removed.append(node)
assert [x.targets[0].id for x in removed] == ['_nsa_d128_factory', 'nsa_shared_chunk128_manual_sync', 'nsa_d128_scheduler']
for node in new.body:
    if isinstance(node, ast.Assign):
        ast.literal_eval(node.value)
assert not any(isinstance(x, ast.keyword) and x.arg == 'compile_flags' for x in ast.walk(new))
old.body = [x for x in old.body[1:] if x not in removed]
new.body = new.body[1:]
assert len(old.body) == len(new.body)
for before, after in zip(old.body, new.body):
    if isinstance(after, ast.FunctionDef) and after.name in ('nsa_shared_chunk128_manual_sync', 'nsa_d128_scheduler'):
        assert len(after.decorator_list) == 1
        dec = after.decorator_list[0]
        assert isinstance(dec, ast.Call) and ast.unparse(dec.func) == 'tilelang.jit'
        assert len(dec.keywords) == 1 and dec.keywords[0].arg == 'pass_configs'
        assert [ast.literal_eval(v) for v in dec.keywords[0].value.values] == [True]*4
        after.decorator_list = []
        if after.name == 'nsa_d128_scheduler':
            after.name = 'nsa_d128_direct_factory'
    assert ast.dump(before) == ast.dump(after), getattr(before, 'name', type(before).__name__)
print(json.dumps(dict(status='PASS', removed_nonliteral_assignments=3,
                     kernel_math_and_dispatch_ast_unchanged=True, custom_compile_flags_removed=True,
                     candidate_lf_sha256=hashlib.sha256(new_source.encode()).hexdigest(),
                     limitation='Checks the observed loader restriction only; not an actual SafeExecutor run or OJ acceptance.')))
