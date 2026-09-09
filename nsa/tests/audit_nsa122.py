"""Audit isolated aligned D64 factories and block-index range equivalence."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
c=(root/'probes/probe_nsa122_d64_block_bounds.py').read_text(encoding='utf-8')
changes=[{"name":"nsa_simplified","edits":[["            i_s = BI[i_b, i_t, i_h, 0] * BS\n            if i_s <= i_t:","            block_id = BI[i_b, i_t, i_h, 0]\n            if block_id >= 0 and block_id <= i_t // BS:\n                i_s = (block_id % (seq_len // BS)) * BS"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Ks[i, j] = K[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)","                for i, j in T.Parallel(BS, D):\n                    Ks[i, j] = K[i_b, i_s + i, i_h, j]"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Vs[i, j] = V[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)","                for i, j in T.Parallel(BS, D):\n                    Vs[i, j] = V[i_b, i_s + i, i_h, j]"]]},{"name":"nsa_simplified_k_fragment","edits":[["            i_s = BI[i_b, i_t, i_h, 0] * BS\n            if i_s <= i_t:","            block_id = BI[i_b, i_t, i_h, 0]\n            if block_id >= 0 and block_id <= i_t // BS:\n                i_s = (block_id % (seq_len // BS)) * BS"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Ks[i, j] = K[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)","                for i, j in T.Parallel(BS, D):\n                    Ks[i, j] = K[i_b, i_s + i, i_h, j]"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Vs[i, j] = V[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)","                for i, j in T.Parallel(BS, D):\n                    Vs[i, j] = V[i_b, i_s + i, i_h, j]"]]},{"name":"nsa_simplified_v_prefetch","edits":[["            i_s = BI[i_b, i_t, i_h, 0] * BS\n            if i_s <= i_t:","            block_id = BI[i_b, i_t, i_h, 0]\n            if block_id >= 0 and block_id <= i_t // BS:\n                i_s = (block_id % (seq_len // BS)) * BS"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Ks[i, j] = K[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)","                for i, j in T.Parallel(BS, D):\n                    Ks[i, j] = K[i_b, i_s + i, i_h, j]"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Vpref[i, j] = V[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)","                for i, j in T.Parallel(BS, D):\n                    Vpref[i, j] = V[i_b, i_s + i, i_h, j]"]]},{"name":"nsa_simplified_k_fragment_v_prefetch","edits":[["            i_s = BI[i_b, i_t, i_h, 0] * BS\n            if i_s <= i_t:","            block_id = BI[i_b, i_t, i_h, 0]\n            if block_id >= 0 and block_id <= i_t // BS:\n                i_s = (block_id % (seq_len // BS)) * BS"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Ks[i, j] = K[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)","                for i, j in T.Parallel(BS, D):\n                    Ks[i, j] = K[i_b, i_s + i, i_h, j]"],["                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        Vpref[i, j] = V[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)","                for i, j in T.Parallel(BS, D):\n                    Vpref[i, j] = V[i_b, i_s + i, i_h, j]"]]}]
gate="        if S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:\n            if fn is nsa_simplified:\n                fn = nsa_simplified_bounded122\n            elif fn is nsa_simplified_k_fragment:\n                fn = nsa_simplified_k_fragment_bounded122\n            elif fn is nsa_simplified_v_prefetch:\n                fn = nsa_simplified_v_prefetch_bounded122\n            elif fn is nsa_simplified_k_fragment_v_prefetch:\n                fn = nsa_simplified_k_fragment_v_prefetch_bounded122\n"
clones=''
for change in changes:
    name=change['name']
    d=p.index('def '+name+'(');a=p.rfind('@tilelang.jit',0,d);b=p.index('\n    return kernel',d)+len('\n    return kernel')
    text=p[a:b].replace('def '+name+'(','def '+name+'_bounded122(')
    for old,new in change['edits']:
        assert text.count(old)==1
        text=text.replace(old,new)
    clones+=text+'\n\n\n'
expected=p.replace('def _get_kernel(',clones+'def _get_kernel(').replace('        kernel = fn(\n',gate+'        kernel = fn(\n')
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(c)==tree(expected)
# All supported aligned lengths: admitted IDs equal valid causal starts; modulo is identity.
for length in range(64,8193,16):
    count=length//16
    for block in range(count):
        assert block%count==block
        assert block*16+15<length
        for t in (block*16, min(length-1,block*16+15),length-1):
            assert (0<=block<=t//16)==(0<=block*16<=t)
    assert not 0<=length<=(length-1)//16
print(json.dumps(dict(audit='PASS',source_sha256=hashlib.sha256(c.encode()).hexdigest(),
    scope='Four cloned D64/G16/BS16 factories selected only for aligned workloads>=4096 queries',
    limitation='No new all-masked-row or negative-index semantics; those are not correctness evidence')))
