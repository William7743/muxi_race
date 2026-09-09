"""Exact AST delta and aligned-block bounds proof checks for D128 probes."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parent=(root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_text(encoding="utf-8")

def expr(s):
    return ast.dump(ast.parse(s,mode="eval").body)

def stripdocs(t):
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return t

class Bounds(ast.NodeTransformer):
    def __init__(self):
        self.guards=self.indices=self.fallbacks=0
    def visit_Assign(self,n):
        if (len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="i_s"
                and ast.dump(n.value)==expr("BI[i_b, i_t, i_h, 0] * BS")):
            self.indices+=1
            return ast.parse("block_id = BI[i_b, i_t, i_h, 0]").body[0]
        return self.generic_visit(n)
    def visit_If(self,n):
        n=self.generic_visit(n)
        if ast.dump(n.test)==expr("i_s <= i_t"):
            self.guards+=1
            n.test=ast.parse("block_id >= 0 and block_id <= i_t // BS",mode="eval").body
            n.body.insert(0,ast.parse("i_s = (block_id % (seq_len // BS)) * BS").body[0])
        if ast.dump(n.test)==expr("i_s >= 0 and i_s + BS <= seq_len"):
            self.fallbacks+=1
            return n.body
        return n

reports=[]
for version,name,bs in ((130,"nsa_d128_scheduler",32),(131,"nsa_chunk128_warp_sync",16)):
    expected=ast.parse(parent)
    old=next(n for n in expected.body if isinstance(n,ast.FunctionDef) and n.name==name)
    transform=Bounds()
    new=transform.visit(old)
    assert transform.indices==transform.guards==1
    assert transform.fallbacks==(2 if version==130 else 1)
    candidate=root/f"probes/probe_nsa{version}_d128_bs{bs}_block_bounds.py"
    actual=ast.parse(candidate.read_text(encoding="utf-8"))
    assert ast.dump(stripdocs(expected))==ast.dump(stripdocs(actual))
    # Dispatch is unchanged and only selects these factories on aligned S1/D128/G16 inputs.
    assert "if S == 1 and D == 128 and seq_len % block_size == 0 and groups >= 16:" in parent
    assert parent.count("fn = nsa_d128_scheduler if H == 1 else nsa_shared_chunk128_manual_sync")==1
    assert parent.count("fn = nsa_chunk128_warp_sync")==1
    checks=0
    for length in (64,96,128,512,1024,1056,8192):
        assert length%bs==0
        for block in [*range(length//bs),length]:
            # All predicate transitions and endpoint tokens for each legal/sentinel block.
            for t in {0,length-1,max(0,block*bs-1),min(length-1,block*bs),min(length-1,block*bs+bs-1)}:
                if t>=length:
                    continue
                original=block*bs<=t
                admitted=block>=0 and block<=t//bs
                assert original==admitted
                if admitted:
                    start=(block%(length//bs))*bs
                    assert start==block*bs and 0<=start<=length-bs
                    assert start+bs-1<length
                checks+=1
    reports.append(dict(version=version,source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        block_size=bs,bounds_cases=checks,static_audit="PASS",gpu_validation="NOT_ASSESSED"))
print(json.dumps(reports))
