"""CPU-only source/host isolation of the v750 E16 Stage2 runtime-M64 selector.

No torch/TileLang import or GPU work. Mocks prove requested dispatch/dataflow,
not numeric correctness, device zero-grid support, or performance.
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
    750: DATA / "probe_v750_v745_e16_stage2_runtime_m64.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)} for v, t in TREE.items()}
TARGET = "num_experts == 32 or (num_experts == 16 and hidden == 2048 and intermediate == 8192)"
OLD_RUNTIME = ("hidden == 7168 and intermediate == 2048 and total_valid_tokens > 0 "
               "and total_padded_tokens > 0 and num_blocks_m > 0")
NEW_RUNTIME = ("((hidden == 7168 and intermediate == 2048) or "
               "(num_experts == 16 and hidden == 2048 and intermediate == 8192)) "
               "and total_valid_tokens > 0 and total_padded_tokens > 0 and num_blocks_m > 0")


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def segment(version, name):
    return ast.get_source_segment(SOURCE[version], FUNCTIONS[version][name])


expected = copy.deepcopy(TREE[745])
expected_functions = {n.name: n for n in expected.body if isinstance(n, ast.FunctionDef)}
choices = [n for n in ast.walk(expected_functions["_get_stage2"])
           if isinstance(n, ast.If) and ast.unparse(n.test) == "num_experts == 32"]
assert len(choices) == 1
choices[0].test = ast.parse(TARGET, mode="eval").body
choices = [n for n in ast.walk(expected_functions["_get_stage2"])
           if isinstance(n, ast.IfExp)
           and ast.dump(n.test) == ast.dump(ast.parse(OLD_RUNTIME, mode="eval").body)]
assert len(choices) == 1
choices[0].test = ast.parse(NEW_RUNTIME, mode="eval").body
for count_name in ("total_padded_tokens", "total_valid_tokens"):
    guards = [n for n in expected_functions["run_kernel"].body if isinstance(n, ast.If)
              and ast.unparse(n.test) == f"num_experts == 32 and {count_name} == 0"]
    assert len(guards) == 1
    guards[0].test = ast.parse(f"({TARGET}) and {count_name} == 0", mode="eval").body
assert ast.dump(expected) == ast.dump(TREE[750]), "Unexpected executable AST change"

expected_text = SOURCE[745][SOURCE[745].index("import torch\n"):]
old = "        if num_experts == 32:\n            # Empty arrays"
new = ("        if num_experts == 32 or (\n"
       "            num_experts == 16 and hidden == 2048 and intermediate == 8192\n"
       "        ):\n            # Empty arrays")
assert expected_text.count(old) == 1
expected_text = expected_text.replace(old, new)
old = ("                    if hidden == 7168 and intermediate == 2048\n"
       "                    and total_valid_tokens > 0 and total_padded_tokens > 0 and num_blocks_m > 0")
new = ("                    if (\n"
       "                        (hidden == 7168 and intermediate == 2048)\n"
       "                        or (num_experts == 16 and hidden == 2048 and intermediate == 8192)\n"
       "                    )\n"
       "                    and total_valid_tokens > 0 and total_padded_tokens > 0 and num_blocks_m > 0")
assert expected_text.count(old) == 1
expected_text = expected_text.replace(old, new)
for count_name in ("total_padded_tokens", "total_valid_tokens"):
    old = f"    if num_experts == 32 and {count_name} == 0:"
    new = "    if (\n        " + TARGET + f"\n    ) and {count_name} == 0:"
    assert expected_text.count(old) == 1
    expected_text = expected_text.replace(old, new)
assert expected_text == SOURCE[750][SOURCE[750].index("import torch\n"):]
for name in FUNCTIONS[745]:
    if name not in ("_get_stage2", "run_kernel"):
        assert segment(745, name) == segment(750, name), name
assert list(FUNCTIONS[745]) == list(FUNCTIONS[750])
compile(SOURCE[750], str(PATHS[750]), "exec")
print("Full-module AST and executable source: only four permitted predicates PASS")
print("Every builder, Stage1, pass, tile, layout, cache key and non-target source unchanged PASS")

runtime = FUNCTIONS[750]["_moe_stage2_runtime_m64_route_bounds"]
swizzles = [n for n in ast.walk(runtime) if isinstance(n, ast.Call) and ast.unparse(n.func) == "T.use_swizzle"]
assert len(swizzles) == 1 and ast.literal_eval(swizzles[0].args[0]) == 2
emitters = [n for n in ast.walk(runtime) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "TensorCoreIntrinEmitter"]
assert len(emitters) == 2
for emitter in emitters:
    settings = {k.arg: k.value for k in emitter.keywords}
    assert ast.literal_eval(settings["k_pack"]) == 1
assert "T.ceildiv(intermediate, be2)" in segment(750, runtime.name)
assert (8192 + 64 - 1) // 64 == 128
zero = FUNCTIONS[750]["_moe_stage2_e32_zero_output"]
zero_kernel = next(n for n in ast.walk(zero) if isinstance(n, ast.With))
for access in [n for n in ast.walk(zero_kernel) if isinstance(n, ast.Subscript)]:
    assert isinstance(access.value, ast.Name) and access.value.id == "out" and isinstance(access.ctx, ast.Store)
print("Inherited Stage2 full/tail dual-B k_pack1/swizzle2; E16 I8192 has 128 K64 tiles; zero-kernel no tensor loads PASS")


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
    exec(compile(module(nodes), "<v750-host>", "exec"), env)

    e32_shape = hidden == 7168 and intermediate == 2048
    e16_shape = experts == 16 and hidden == 2048 and intermediate == 8192
    managed = experts == 32 or (version == 750 and e16_shape)
    expected_count = 0 if managed and padded == 0 else (1 if managed and valid == 0 else 2)
    if managed:
        use_runtime_s2 = ((experts == 32 and e32_shape) or (version == 750 and e16_shape))
        use_runtime_s2 = use_runtime_s2 and padded > 0 and blocks > 0
        expected_s2 = ("_moe_stage2_e32_zero_output" if valid == 0 else
                       "_moe_stage2_runtime_m64_route_bounds" if use_runtime_s2 else
                       "_moe_stage2_fast_bfrag_prefetch_route_bounds")
    else:
        expected_s2 = "_moe_stage2_fast_bfrag_prefetch" if experts in (16, 64) else "_moe_stage2_fast"
    use_runtime_s1 = experts == 32 and e32_shape and padded > 0 and blocks > 0
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



# Include both cross-shape traps: E32/H2048/I8192 must NOT gain runtime-M64;
# E16/H7168/I2048 must NOT be managed despite matching E32's inner dimensions.
shapes = ((1, 512, 256), (8, 7168, 2048),
          (16, 2048, 8192), (16, 4096, 8192), (16, 2048, 4096), (16, 7168, 2048),
          (32, 7168, 2048), (32, 2048, 8192), (32, 4096, 2048), (32, 7168, 1024),
          (64, 7168, 2048), (64, 2048, 8192), (64, 4096, 2048), (64, 7168, 1024))
count = 0
for (experts, hidden, intermediate), valid, padded, blocks, dtype in itertools.product(
        shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    args = experts, hidden, intermediate, valid, padded, blocks, dtype
    results = {v: check_host(v, *args) for v in PATHS}
    target = experts == 16 and hidden == 2048 and intermediate == 8192
    if not target:
        assert results[745] == results[750]
    else:
        assert results[745][0] == 2
        expected = 0 if padded == 0 else (1 if valid == 0 else 2)
        assert results[750][0] == expected
        if expected == 2:
            assert results[745][1][0] == results[750][1][0], "Stage1 must stay identical"
            assert results[750][1][-1][1][:6] == (hidden, intermediate, experts, padded, valid, blocks)
    count += 1
print(f"Host {count} combinations x 2 fresh calls x v745/v750 PASS")
print("Exact E16 new 0/1/2 launch requests; E32 cross-shape/neighbors/E64/other paths unchanged PASS")
print("LIMIT: mocks do not prove zero-grid GPU support, malformed map validity, bitwise equality or speed.")
for version in PATHS:
    print(f"v{version} SHA256 {hashlib.sha256(PATHS[version].read_bytes()).hexdigest()}")
