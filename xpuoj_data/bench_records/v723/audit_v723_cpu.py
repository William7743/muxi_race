import ast, copy, itertools, pathlib, struct, types
root = pathlib.Path(__file__).resolve().parents[2]
base_path = root / "probe_v720_v719_e16_stage2_bfrag_only.py"
probe_path = root / "probe_v723_v720_e32_route_load_bounds.py"
base = ast.parse(base_path.read_text(encoding="utf-8"))
probe = ast.parse(probe_path.read_text(encoding="utf-8"))
compile(probe, str(probe_path), "exec")
old = {n.name:n for n in base.body if isinstance(n, ast.FunctionDef)}
new = {n.name:n for n in probe.body if isinstance(n, ast.FunctionDef)}
changed = {"_get_stage2", "run_kernel"}
for name, node in old.items():
    if name not in changed:
        assert ast.dump(node, include_attributes=False) == ast.dump(new[name], include_attributes=False), name
clone = copy.deepcopy(new["_moe_stage2_fast_bfrag_prefetch_route_bounds"])
clone.name = "_moe_stage2_fast_bfrag_prefetch"
clone.body[0] = copy.deepcopy(old[clone.name].body[0])
class UndoClamp(ast.NodeTransformer):
    count = 0
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "routed_expert_weights":
            assert ast.unparse(node.slice) == "T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))"
            self.count += 1
            node.slice = ast.parse("raw_start + token_offset + i", mode="eval").body
        return node
undo = UndoClamp()
clone = undo.visit(clone)
assert undo.count == 2
assert ast.dump(clone, include_attributes=False) == ast.dump(old[clone.name], include_attributes=False)
print("AST: all existing builders/helpers unchanged except E32 dispatch/run guards; cloned builder differs only at 2 route indices.")

zero_node = copy.deepcopy(new["_moe_stage2_e32_zero_output"])
zero_inner = next(n for n in zero_node.body if isinstance(n, ast.FunctionDef))
assert len(zero_inner.args.args) == 8
assert ast.dump(zero_node.args, include_attributes=False) == ast.dump(old["_moe_stage2_fast_bfrag_prefetch"].args, include_attributes=False)
assert ast.dump(zero_inner.args, include_attributes=False) == ast.dump(next(n for n in old["_moe_stage2_fast_bfrag_prefetch"].body if isinstance(n, ast.FunctionDef)).args, include_attributes=False)
assert not any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) for s in zero_inner.body for n in ast.walk(s))
class KernelAsLoop(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        assert len(node.items) == 1
        item = node.items[0]
        assert ast.unparse(item.context_expr.func) == "T.Kernel"
        return ast.copy_location(ast.For(target=item.optional_vars, iter=item.context_expr, body=node.body, orelse=[]), node)
zero_node.decorator_list = []
zero_node = KernelAsLoop().visit(zero_node)
class Poison:
    def __getitem__(self, key): raise AssertionError("zero kernel read an input")
class ZeroOut:
    def __init__(self, rows, cols): self.rows,self.cols,self.writes = rows,cols,{}
    def __setitem__(self, key, value):
        i,j=key
        assert 0 <= i < self.rows and 0 <= j < self.cols, key
        assert value == 0 and key not in self.writes
        self.writes[key] = value
zero_t = types.SimpleNamespace(float16="float16", int32="int32",
    Tensor=lambda *args: object, prim_func=lambda fn: fn,
    ceildiv=lambda a,b:(a+b-1)//b,
    Kernel=lambda x,y,threads: itertools.product(range(x),range(y)),
    Parallel=lambda x,y: itertools.product(range(x),range(y)))
env_zero={"T":zero_t}
exec(compile(ast.fix_missing_locations(ast.Module(body=[zero_node],type_ignores=[])),"zero_cpu","exec"),env_zero)
zero_cases=0
for rows,cols,wd in itertools.product((1,127,128,129,256),(1,127,128,129),("float16","float32")):
    fn=env_zero[zero_node.name](cols,2048,32,rows,0,0,128,128,64,256,wd)
    out=ZeroOut(rows,cols)
    fn(*([Poison()]*7),out)
    assert len(out.writes)==rows*cols
    zero_cases+=1
print(f"Zero builder: identical signature; no input-load AST; {zero_cases} CPU-interpreted tiles cover output exactly, including partial tails and empty block map.")

dispatch_names={"_pick_tiles","_get_stage1","_get_stage2","_get_workspace","run_kernel"}
dispatch_nodes=[copy.deepcopy(n) for n in probe.body if isinstance(n,ast.FunctionDef) and n.name in dispatch_names]
class Tensor:
    def __init__(self,shape,dtype="float16",device="cpu"):
        self.shape,self.dtype,self.device=shape,dtype,device
launches=[]; builds=[]
def builder_stub(name):
    def build(*args):
        builds.append((name,args))
        def launch(*tensors): launches.append((name,tensors))
        return launch
    return build
torch=types.SimpleNamespace(float16="float16",float32="float32",
    empty=lambda shape,device,dtype:Tensor(shape,dtype,device))
env={"torch":torch,"T":types.SimpleNamespace(float16="float16",float32="float32"),
     "_KERNEL_CACHE":{},"_WORKSPACE_CACHE":{}}
for name in new:
    if name.startswith("_moe_"): env[name]=builder_stub(name)
exec(compile(ast.fix_missing_locations(ast.Module(body=dispatch_nodes,type_ignores=[])),"dispatch_cpu","exec"),env)
dispatch_cases=0
for experts,valid,padded,wd in itertools.product((8,16,32,64),(0,7),(0,256),("float16","float32")):
    launches.clear(); builds.clear(); env["_KERNEL_CACHE"].clear(); env["_WORKSPACE_CACHE"].clear()
    args=[Tensor((padded,7168)),Tensor((experts,2048,7168)),Tensor((experts,2048,7168)),
          Tensor((experts,7168,2048)),Tensor((valid,),wd),Tensor((experts,),"int32"),
          Tensor((experts+1,),"int32"),Tensor((experts+1,),"int32"),
          Tensor((0 if valid==0 else 2,),"int32"),Tensor((padded,7168))]
    env["run_kernel"](*args)
    if experts==32 and padded==0:
        assert launches==[] and builds==[] and env["_WORKSPACE_CACHE"]=={}
        expected=[]
    elif experts==32 and valid==0:
        expected=["_moe_stage2_e32_zero_output"]
    else:
        s1="_moe_stage1_prefetch_giu_merge" if experts in (32,64) else "_moe_stage1_prefetch"
        s2="_moe_stage2_fast_bfrag_prefetch_route_bounds" if experts==32 else (
            "_moe_stage2_fast_bfrag_prefetch" if experts in (16,64) else "_moe_stage2_fast")
        expected=[s1,s2]
    assert [n for n,a in launches]==expected,(experts,valid,padded,wd,launches)
    if expected:
        assert builds[-1][1][-1]==wd
        assert launches[-1][1][-1] is args[-1]
    num_builds=len(builds)
    env["run_kernel"](*args)
    assert [n for n,a in launches]==expected*2
    assert len(builds)==num_builds
    dispatch_cases+=1
print(f"Dispatch: {dispatch_cases} expert/empty/FP16/FP32 cases pass; second call reuses JIT but always relaunches current work.")

def f32(x): return struct.unpack("<f",struct.pack("<f",x))[0]
def f16(x): return struct.unpack("<e",struct.pack("<e",x))[0]
groups_list=[
 [0]*32,
 [1]+[0]*31,
 [0]*31+[1],
 [128]+[0]*31,
 [0]*30+[128,1],
 [64,220]*16,
 [127,128,129,0]*8,
 [0,0,3,256]+[0]*28,
]
checked_valid=checked_padding=0
for groups,wd in itertools.product(groups_list,("float16","float32")):
    total=sum(groups)
    if total==0: continue
    cast=f16 if wd=="float16" else f32
    weights=[cast(((k*17)%91-45)/19) for k in range(total)]
    raw_start=0
    for size in groups:
        padded=((size+127)//128)*128
        for token_offset in range(0,padded,128):
            actual=max(0,min(128,size-token_offset))
            for i in range(128):
                index=raw_start+token_offset+i
                bounded=max(0,min(index,total-1))
                assert 0<=bounded<total
                if i<actual:
                    assert bounded==index
                    # FP32 accumulator/multiply and final FP16 store remain exactly at the same point.
                    acc=f32(((index*13)%227-113)/31)
                    old_value=struct.pack("<e",f32(acc*f32(weights[index])))
                    new_value=struct.pack("<e",f32(acc*f32(weights[bounded])))
                    assert old_value==new_value
                    checked_valid+=1
                else:
                    # A speculative weight read may occur, but output is an explicit zero.
                    _=weights[bounded]
                    assert struct.pack("<e",0.0)==b"\x00\x00"
                    checked_padding+=1
        raw_start+=size
print(f"Address/rounding: {checked_valid} valid rows preserve exact FP16 bytes across FP16/FP32 route weights; {checked_padding} padding rows keep bounded speculative loads and explicit zero stores.")
print("PASS: syntax, AST isolation, same-signature no-load zero kernel, CPU dispatch/cache, empty/partial/full-group indices, FP16/FP32 epilogue equivalence. No GPU or OJ performed.")
