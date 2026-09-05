# v746: extend v745 runtime-M64 Stage1 to E64

## Status and frozen identity

**Local E64 candidate; not recommended for OJ yet.** Python/Ruff and independent
source/CPU host checks passed, followed by E64 TileLang compilation and full
generated-source review. The identical Stage1 component passed the v748 edge
helper below. Independent normal-entry correctness/timing has now completed
on two fixtures; no OJ submission or score is claimed for v746. Its source
header retains the creation-time status to preserve the exact tested hash;
this README and the linked batch record give the updated test status.

- Candidate: [probe_v746_v745_e64_stage1_runtime_m64.py](../../probe_v746_v745_e64_stage1_runtime_m64.py).
- Candidate SHA256: `9cd17d1b2b8e02fd59fb277d602e9ad03e654b932aa536b43211075dca7e3416`.
- Frozen v745 cloned: `12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96`.
- While auditing, the main thread updated only v745's header, giving
  `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465`.
  Reattaching the captured original header to its current executable body
  independently reproduces the frozen v745 SHA exactly. No rebase occurred;
  candidate executable-text equality also passed against the current body.

A local filename/version-reference scan found no v746 before creation. Only
this probe and this README were created: parent probes, submission.py,
OPTIMIZATION_LOG, GPU/SSH and Git were not modified. These two files remain
outside the current v744/v745 commit for a later test cycle.

## Exactly one executable change

Inside `_get_stage1`, replace the existing outer selector:

```python
if num_experts == 32
```

with:

```python
if num_experts in (32, 64)
```

The inner condition remains H7168/I2048 with padded>0 and blocks>0. Its
fallback remains `_moe_stage1_prefetch_giu_merge`. The original outer fallback
is intentionally retained, including its now-redundant E64 test, so no other
executable text changes.

| Selection | v745 | v746 |
| --- | --- | --- |
| E32/H7168/I2048, positive padded/blocks | Runtime M128/M64 GIU | unchanged |
| E64/H7168/I2048, positive padded/blocks | Original terminal-K GIU | Existing runtime M128/M64 GIU |
| E32/E64 H or I neighbors, zero padded/blocks at getter level | Original terminal-K GIU | unchanged |
| E16/E1/E8/other experts | Previous selection | unchanged |
| Stage2 and host empty-input handling | v745 | byte-identical |

No new total-valid-token condition is added to the Stage1 getter. The host
still skips Stage1 for E32 empty routes, but gains no E64 equivalent.

## E64 computation preserved; hypothesis still untested

The prior E64 path actually selects `_moe_stage1_prefetch_giu_merge`, not the
retained `_moe_stage1_prefetch_giu_merge_v527`. Its last four statements
(clear Gate, clear Up, guarded terminal-K computation, valid-only epilogue)
are exactly the runtime builder's full-branch AST. Builder arguments and
passes match, and every builder body is byte-identical to v745.

The existing runtime builder uses full M128 for rows>64, M64 for1..64, and
no workspace stores for zero rows. It keeps N128/K64,256 threads, T.gemm
Square policy, k_pack2, vecSize4 and current-K Gate/Input/Up loading. H7168
uses steady K0..110 plus terminal K111, including the unchanged missing
terminal end-K barrier. SwiGLU order and FP16 workspace output are unchanged.
The swizzle expression `3 if num_experts == 32 else 2` evaluates to2 for E64
in both old and runtime builders; E32 stays swizzle3.

Shared remains one128x64 Input tile and one reusable128x64 weight tile
(32 KiB), with the tail using the first64 Input rows. Full/tail Gate/Up
accumulator fragments are distinct. Physical register allocation, occupancy
and actual E64 view lowering have not been measured or verified here.

The possible benefit is fewer Input rows and Gate/Up MMA rows on short E64
blocks. Weight columns/loads are unchanged. A distribution with few short
blocks, branch code or register allocation could erase the benefit; the E32
result does not establish E64 performance. No new buffer, global workspace,
launch, async/BSM/pipeline, extern or historical result replay is introduced.

## Inherited limits are not fixed by this candidate

Stage2 and its dispatcher remain byte-identical to v745. E64 still selects
`_moe_stage2_fast_bfrag_prefetch` with the original raw-route addressing.
This does **not** extend E32's route-load clamps, empty-route zero kernel or
zero-padded early return to E64. Do not describe it as all-shape memory safe.

Stage1 leaves invalid workspace rows untouched. Existing Stage2 retains its
prior padded-output responsibility and risks. Empty-route/padded/block mock
cases below establish host dispatch and arguments only, not safety of
zero-length device access, zero-grid launches, invalid metadata or zero
dimensions. Supported positive inputs still execute Stage1 then Stage2 with
fresh current-input work; only allocation/JIT callables are reused.

## Completed static and host checks

- Entire module AST equals v745 after the one selector edit and header changes.
- Executable text from `import torch` to EOF is the exact expected replacement;
  every function except `_get_stage1` is text-identical.
- Old E64 full body, arguments/passes, all GEMM keyword choices, k_pack2 and
  E64 swizzle2 were independently checked.
- 216 host combinations, two fresh input sets each: E1/E8/E16/E32/E64 plus
  E32/E64 H/I neighbors, FP16/FP32 routes, raw0/1/129, padded0/256, blocks0/2.
  Only the intended E64 Stage1 callable changes; arguments, launch requests
  and workspace/JIT-only reuse match the parent.
- Python compilation and Ruff passed. The initial executable-text check caught
  an extra trailing blank line during creation; removal preceded the frozen
  hash above, after which exact-text isolation passed.

The standard-library audit below was run from the repository root. It reuses
only the existing v743 audit's host instrumentation definitions, not its
candidate-specific expectations. It imports no torch/TileLang and writes no
files or device state.

```python
import ast,copy,hashlib,itertools,types
from pathlib import Path
p=Path('xpuoj_data')
paths={745:p/'probe_v745_v743_e32_stage1_runtime_m64.py',746:p/'probe_v746_v745_e64_stage1_runtime_m64.py'}
source={v:f.read_text(encoding='utf-8') for v,f in paths.items()}
TREE={v:ast.parse(s) for v,s in source.items()}
FUNCTIONS={v:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)} for v,t in TREE.items()}
expected=copy.deepcopy(TREE[745]); getter=next(n for n in expected.body if isinstance(n,ast.FunctionDef) and n.name=='_get_stage1')
choices=[n for n in ast.walk(getter) if isinstance(n,ast.IfExp) and ast.unparse(n.test)=='num_experts == 32']; assert len(choices)==1
choices[0].test=ast.parse('num_experts in (32, 64)',mode='eval').body
assert ast.dump(expected)==ast.dump(TREE[746])
for name in FUNCTIONS[745]:
    if name!='_get_stage1': assert ast.get_source_segment(source[745],FUNCTIONS[745][name])==ast.get_source_segment(source[746],FUNCTIONS[746][name]),name
old=ast.get_source_segment(source[745],FUNCTIONS[745]['_get_stage1']); new=old.replace('if num_experts == 32','if num_experts in (32, 64)')
assert source[745][source[745].index('import torch\n'):].replace(old,new)==source[746][source[746].index('import torch\n'):]
base=FUNCTIONS[745]['_moe_stage1_prefetch_giu_merge']; runtime=FUNCTIONS[746]['_moe_stage1_runtime_m64_giu_merge']
assert ast.dump(base.args)==ast.dump(runtime.args)
assert [ast.dump(x) for x in base.decorator_list]==[ast.dump(x) for x in runtime.decorator_list]
kb=next(n for n in ast.walk(base) if isinstance(n,ast.With)); kr=next(n for n in ast.walk(runtime) if isinstance(n,ast.With))
branch=next(n for n in kr.body if isinstance(n,ast.If) and ast.unparse(n.test)=='actual_rows > tail_m')
assert [ast.dump(n) for n in kb.body[-4:]]==[ast.dump(n) for n in branch.body]
for f in [base,runtime]:
    assert any(isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='gu_k_pack' and ast.unparse(n.value)=='2' for n in f.body)
    sw=[n for n in ast.walk(f) if isinstance(n,ast.Call) and ast.unparse(n.func)=='T.use_swizzle']; assert len(sw)==1
    assert ast.unparse(sw[0].args[0])=='3 if num_experts == 32 else 2'
    assert eval(compile(ast.Expression(sw[0].args[0]),'<swizzle>','eval'),{}, {'num_experts':64})==2
    for call in [n for n in ast.walk(f) if isinstance(n,ast.Call) and ast.unparse(n.func)=='T.gemm']:
        assert {k.arg:ast.unparse(k.value) for k in call.keywords}=={'transpose_B':'True','policy':'T.GemmWarpPolicy.Square','k_pack':'gu_k_pack'}
print('Whole-module AST and exact executable text: only Stage1 E32 -> E32/E64 selector PASS')
print('All builders/Stage2/host unchanged; E64 old terminal-GIU, full runtime body exact, passes/k_pack2/Square/swizzle2 PASS')
helper=ast.parse((p/'bench_records/v743/audit_v743_cpu.py').read_text(encoding='utf-8'))
def module(ns): return ast.Module(body=ns,type_ignores=[])
selected=[n for n in helper.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Tensor','host_mock')]
exec(compile(module(selected),'<host helper>','exec'),globals())
shapes=((1,512,256),(8,7168,2048),(16,2048,8192),(32,7168,2048),(32,4096,2048),(32,7168,1024),(64,7168,2048),(64,4096,2048),(64,7168,1024))
count=0
for (e,h,i),valid,pad,blocks,dtype in itertools.product(shapes,(0,1,129),(0,256),(0,2),('float16','float32')):
    old=host_mock(745,e,h,i,valid,pad,blocks,dtype); new=host_mock(746,e,h,i,valid,pad,blocks,dtype)
    expected=copy.deepcopy(old)
    if e==64 and h==7168 and i==2048 and pad>0 and blocks>0:
        assert expected[0][0]=='_moe_stage1_prefetch_giu_merge'
        expected[0]=('_moe_stage1_runtime_m64_giu_merge',expected[0][1])
    assert new==expected,(e,h,i,valid,pad,blocks,dtype)
    count+=1
print('Host',count,'cases x 2 fresh calls: exact target/fallback, args, launches, dtypes, JIT/workspace-only reuse PASS')
for v,f in paths.items():
    compile(source[v],str(f),'exec'); print(v,'SHA256',hashlib.sha256(f.read_bytes()).hexdigest())
print('LIMIT: no new E64 raw clamp/empty guard; no GPU codegen/precision/bitwise/timing assertion')
```

## Completed E64 compile and shared-component boundary checks

The captured [E64 Stage1 source](codex_e64_745_746_stage1_codegen.log) and
[independent audit](CODEGEN_AUDIT.md) verify complete generated bodies against
the E32 originals, changing only diagnostic kernel names and swizzle3→2.
The runtime branch has25102 characters versus12945 for the original GIU;
both retain32KiB shared. Static source size is not a physical-resource or
performance estimate.

The v748 helper subsequently tested this exact Stage1 builder/selection on
E64 raw4746/padded9216 with empty/full/tail CTAs and two fresh X/route sets:
valid Stage1 rows were actually bitwise equal to original GIU, padding stayed
NaN. See [shared-component results and limitations](../v747_v748/README.md).
Its final chain uses v748's clamped runtime Stage2, not v746's original E64
Stage2; do not call that a standalone v746 normal-entry correctness test.
Frozen sources remain unchanged for the separate four-candidate entry batch.

## Completed normal-entry batch

Both random-valued routing fixtures passed three repeated NaN-poisoned
full-chain/real-entry tolerance checks for each independent candidate, with
one warmup, one measured entry call per round and four forward/reverse rounds.
v746 entry medians were8.807552ms vs v7458.907008ms on alternating routing
(4/4 paired faster), and8.187264ms vs8.222464ms on synthetic routing (3/4).
These are about1.12%/0.43% lower local median latencies, not an OJ result or
bitwise proof. All raw samples/reference limitations are in the
[two-fixture batch record](../v747_v748/README.md). The combination v748 is
selected for the next manual OJ test; v746 remains an isolation control.
