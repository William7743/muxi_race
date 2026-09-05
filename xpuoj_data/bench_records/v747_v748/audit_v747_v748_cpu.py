"""CPU-only source/host isolation of E64 Stage2 and combined-tail candidates.

No torch/TileLang import, GPU execution, input values or performance claims.
Mock launch requests are not proof of device acceptance of zero-grid launches.
"""

import ast
import copy
import hashlib
import itertools
from pathlib import Path
import types


DATA = Path(__file__).resolve().parents[2]
PATHS = {
    745: DATA / "probe_v745_v743_e32_stage1_runtime_m64.py",
    747: DATA / "probe_v747_v745_e64_stage2_runtime_m64.py",
    748: DATA / "probe_v748_v747_e64_stage1_runtime_m64.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)} for v, t in TREE.items()}
TARGET = "num_experts == 32 or (num_experts == 64 and hidden == 7168 and intermediate == 2048)"


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def segment(version, name):
    return ast.get_source_segment(SOURCE[version], FUNCTIONS[version][name])


for version in (747, 748):
    expected = copy.deepcopy(TREE[745])
    expected_functions = {n.name: n for n in expected.body if isinstance(n, ast.FunctionDef)}
    choices = [n for n in ast.walk(expected_functions["_get_stage2"])
               if isinstance(n, ast.If) and ast.unparse(n.test) == "num_experts == 32"]
    assert len(choices) == 1
    choices[0].test = ast.parse(TARGET, mode="eval").body
    for count in ("total_padded_tokens", "total_valid_tokens"):
        guards = [n for n in expected_functions["run_kernel"].body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == f"num_experts == 32 and {count} == 0"]
        assert len(guards) == 1
        guards[0].test = ast.parse(f"({TARGET}) and {count} == 0", mode="eval").body
    if version == 748:
        choices = [n for n in ast.walk(expected_functions["_get_stage1"])
                   if isinstance(n, ast.IfExp) and ast.unparse(n.test) == "num_experts == 32"]
        assert len(choices) == 1
        choices[0].test = ast.parse("num_experts in (32, 64)", mode="eval").body
    assert ast.dump(expected) == ast.dump(TREE[version]), f"v{version}: unexpected AST change"

    expected_text = SOURCE[745][SOURCE[745].index("import torch\n"):]
    getter = segment(745, "_get_stage2")
    changed = getter.replace("if num_experts == 32:", "if num_experts == 32 or (\n"
                             "            num_experts == 64 and hidden == 7168 and intermediate == 2048\n        ):")
    expected_text = expected_text.replace(getter, changed)
    for count in ("total_padded_tokens", "total_valid_tokens"):
        old = f"    if num_experts == 32 and {count} == 0:"
        new = "    if (\n        " + TARGET + f"\n    ) and {count} == 0:"
        assert expected_text.count(old) == 1
        expected_text = expected_text.replace(old, new)
    if version == 748:
        old = segment(745, "_get_stage1")
        expected_text = expected_text.replace(old, old.replace("if num_experts == 32", "if num_experts in (32, 64)"))
    assert expected_text == SOURCE[version][SOURCE[version].index("import torch\n"):]
    allowed = {"_get_stage2", "run_kernel"} | ({"_get_stage1"} if version == 748 else set())
    for name in FUNCTIONS[745]:
        if name not in allowed:
            assert segment(745, name) == segment(version, name), (version, name)
    compile(SOURCE[version], str(PATHS[version]), "exec")
print("Full module AST/executable text: only three E64-exact-shape Stage2/empty guards, plus v748 Stage1 selector PASS")

FROZEN746 = DATA / "probe_v746_v745_e64_stage1_runtime_m64.py"
assert hashlib.sha256(FROZEN746.read_bytes()).hexdigest() == "9cd17d1b2b8e02fd59fb277d602e9ad03e654b932aa536b43211075dca7e3416"
tree746 = ast.parse(FROZEN746.read_text(encoding="utf-8"))
getter746 = next(n for n in tree746.body if isinstance(n, ast.FunctionDef) and n.name == "_get_stage1")
assert ast.dump(getter746) == ast.dump(FUNCTIONS[748]["_get_stage1"])
assert segment(747, "_get_stage1") == segment(745, "_get_stage1")

for name in ("_moe_stage1_runtime_m64_giu_merge", "_moe_stage2_runtime_m64_route_bounds"):
    builder = FUNCTIONS[748][name]
    swizzle = [n for n in ast.walk(builder) if isinstance(n, ast.Call) and ast.unparse(n.func) == "T.use_swizzle"]
    assert len(swizzle) == 1
    assert eval(compile(ast.Expression(swizzle[0].args[0]), "<swizzle>", "eval"), {}, {"num_experts": 64}) == 2
zero = FUNCTIONS[748]["_moe_stage2_e32_zero_output"]
zero_kernel = next(n for n in ast.walk(zero) if isinstance(n, ast.With))
for access in [n for n in ast.walk(zero_kernel) if isinstance(n, ast.Subscript)]:
    assert isinstance(access.value, ast.Name) and access.value.id == "out" and isinstance(access.ctx, ast.Store)
print("v747 Stage1 unchanged; v748 Stage1 exactly v746; all builder bodies/passes/layouts inherited; E64 swizzle2 and zero-kernel no tensor loads PASS")


class Tensor:
    def __init__(self, shape, dtype="float16", device="mock:0"):
        self.shape, self.dtype, self.device = shape, dtype, device


def check_host(version, experts, hidden, intermediate, valid, padded, blocks, route_dtype):
    built, launched, workspaces = [], [], []

    def empty(shape, *, device, dtype):
        result = Tensor(shape, dtype, device)
        workspaces.append(result)
        return result

    def builder(name):
        def build(*parameters):
            built.append((name, parameters))

            def launch(*inputs):
                launched.append((name, inputs))

            return launch

        return build

    env = {"torch": types.SimpleNamespace(float16="float16", float32="float32", empty=empty),
           "T": types.SimpleNamespace(float16="float16", float32="float32"),
           "_KERNEL_CACHE": {}, "_WORKSPACE_CACHE": {}}
    for name in FUNCTIONS[version]:
        if name.startswith("_moe_"):
            env[name] = builder(name)
    names = {"_pick_tiles", "_get_stage1", "_get_stage2", "_get_workspace", "run_kernel"}
    nodes = [copy.deepcopy(n) for n in TREE[version].body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(module(nodes), "<v747-v748-host>", "exec"), env)

    exact_shape = hidden == 7168 and intermediate == 2048
    managed = experts == 32 or (version in (747, 748) and experts == 64 and exact_shape)
    expected_count = 0 if managed and padded == 0 else (1 if managed and valid == 0 else 2)
    if managed:
        expected_s2 = ("_moe_stage2_e32_zero_output" if valid == 0 else
                       "_moe_stage2_runtime_m64_route_bounds" if exact_shape and padded > 0 and blocks > 0 else
                       "_moe_stage2_fast_bfrag_prefetch_route_bounds")
    else:
        expected_s2 = "_moe_stage2_fast_bfrag_prefetch" if experts in (16, 64) else "_moe_stage2_fast"
    use_runtime_s1 = (experts == 32 or (version == 748 and experts == 64)) and exact_shape and padded > 0 and blocks > 0
    expected_s1 = ("_moe_stage1_runtime_m64_giu_merge" if use_runtime_s1 else
                   "_moe_stage1_prefetch_giu_merge" if experts in (32, 64) else "_moe_stage1_prefetch")
    prior_inputs = None
    for repeat in range(2):
        inputs = (Tensor((padded, hidden)), Tensor((experts, intermediate, hidden)),
                  Tensor((experts, intermediate, hidden)), Tensor((experts, hidden, intermediate)),
                  Tensor((valid,), route_dtype), Tensor((experts,), "int32"),
                  Tensor((experts + 1,), "int32"), Tensor((experts + 1,), "int32"),
                  Tensor((blocks,), "int32"), Tensor((padded, hidden)))
        if prior_inputs is not None:
            assert all(x is not y for x, y in zip(prior_inputs, inputs))
        before = len(launched)
        env["run_kernel"](*inputs)
        now = launched[before:]
        assert len(now) == expected_count and len(built) == expected_count
        if not expected_count:
            assert not workspaces
        else:
            assert len(workspaces) == 1
            workspace = workspaces[0]
            assert workspace.shape == (padded, intermediate)
            assert now[-1][0] == expected_s2 and built[-1][1][-1] == route_dtype
            assert now[-1][1] == (workspace,) + tuple(inputs[i] for i in (3, 4, 5, 6, 7, 8, 9))
            if expected_count == 2:
                assert now[0][0] == expected_s1
                assert now[0][1] == tuple(inputs[i] for i in (0, 1, 2, 5, 7, 8)) + (workspace,)
        assert len(launched) == (repeat + 1) * expected_count
        prior_inputs = inputs
    return expected_count, built


shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192),
          (32, 7168, 2048), (32, 4096, 2048), (32, 7168, 1024),
          (64, 7168, 2048), (64, 4096, 2048), (64, 7168, 1024))
count = 0
for (experts, hidden, intermediate), valid, padded, blocks, dtype in itertools.product(
        shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    args = experts, hidden, intermediate, valid, padded, blocks, dtype
    results = {v: check_host(v, *args) for v in PATHS}
    target = experts == 64 and hidden == 7168 and intermediate == 2048
    if not target:
        assert results[745] == results[747] == results[748]
    else:
        assert results[745][0] == 2
        expected = 0 if padded == 0 else (1 if valid == 0 else 2)
        assert results[747][0] == results[748][0] == expected
        if expected == 2:
            assert results[747][1][-1] == results[748][1][-1]
    count += 1
print(f"Host {count} combinations x 2 fresh calls x v745/v747/v748 PASS: exact E64 new 0/1/2 launches; E32/neighbors/other paths unchanged")
print("LIMIT: mocks prove host requests and cache dataflow, not GPU empty-array safety, zero-grid execution, numerical equality or speed.")
for version in PATHS:
    print(f"v{version} SHA256 {hashlib.sha256(PATHS[version].read_bytes()).hexdigest()}")
