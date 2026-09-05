import ast
import hashlib
from pathlib import Path
import sys
import types


DATA = Path(__file__).resolve().parents[2]
ROOT = DATA.parent
FILES = {
    714: DATA / "probe_v714_v713_e32_stage2_bfrag_only.py",
    715: DATA / "probe_v715_v713_e64_stage1_giu_merge_only.py",
    716: DATA / "probe_v716_v714_e64_stage1_giu_merge_only.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


added_name = "_moe_stage1_prefetch_giu_merge_v527"
assert set(FUNCS[716]) == set(FUNCS[714]) | {added_name}
for name in FUNCS[714]:
    if name != "_get_stage1":
        assert ast.dump(FUNCS[716][name]) == ast.dump(FUNCS[714][name]), name
        assert segment(716, name) == segment(714, name), name
for name in (added_name, "_get_stage1"):
    assert ast.dump(FUNCS[716][name]) == ast.dump(FUNCS[715][name]), name
    assert segment(716, name) == segment(715, name), name

# Import statements, global caches, and all non-function executable code are identical.
other = lambda version: ast.dump(ast.Module(
    body=[node for node in TREES[version].body if not isinstance(node, ast.FunctionDef)],
    type_ignores=[],
))
assert other(716) == other(714)

# Remove the transplanted builder and restore just its dispatch: the normalized
# complete source after the header must be byte-for-byte v714, not just AST-equivalent.
restored = SOURCES[716].replace(segment(716, added_name) + "\n\n\n", "")
restored = restored.replace(segment(716, "_get_stage1"), segment(714, "_get_stage1"))
assert restored[restored.index("import torch"):] == SOURCES[714][SOURCES[714].index("import torch"):]


class Tensor:
    def __init__(self, shape, dtype, name):
        self.shape = shape
        self.dtype = dtype
        self.device = "mock:0"
        self.name = name


def load_mock(version):
    compiled = []
    launches = []
    allocated = []
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    def empty(shape, device, dtype):
        tensor = Tensor(shape, dtype, "workspace")
        assert device == "mock:0"
        allocated.append(tensor)
        return tensor
    torch.empty = empty
    lang = types.ModuleType("tilelang.language")
    lang.float16 = "float16"
    lang.float32 = "float32"
    tilelang = types.ModuleType("tilelang")
    tilelang.PassConfigKey = types.SimpleNamespace(TL_DISABLE_WARP_SPECIALIZED="disable_ws")
    def jit(*, pass_configs):
        def decorate(original):
            def builder(*args):
                compiled.append((original.__name__, args, dict(pass_configs)))
                def launch(*inputs):
                    launches.append((original.__name__, args, inputs))
                return launch
            return builder
        return decorate
    tilelang.jit = jit
    intrinsics = types.ModuleType("tilelang.maca.intrinsics")
    intrinsics.TensorCoreIntrinEmitter = object
    intrinsics.make_mma_swizzle_layout = object
    replacements = {"torch": torch, "tilelang": tilelang,
                    "tilelang.language": lang, "tilelang.maca.intrinsics": intrinsics}
    saved = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        namespace = {}
        exec(compile(TREES[version], str(FILES[version]), "exec"), namespace)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return namespace, compiled, launches, allocated


def check_mock(version, experts, routed_dtype):
    namespace, compiled, launches, allocated = load_mock(version)
    hidden = 2048 if experts == 16 else 7168 if experts in (32, 64) else 512
    intermediate = 8192 if experts == 16 else 2048 if experts in (32, 64) else 256
    padded = experts * 256
    valid = experts * 142
    sets = []
    for repeat in range(2):
        specs = [((padded, hidden), "float16"),
                 ((experts, intermediate, hidden), "float16"),
                 ((experts, intermediate, hidden), "float16"),
                 ((experts, hidden, intermediate), "float16"),
                 ((valid,), routed_dtype), ((experts,), "int32"),
                 ((experts + 1,), "int32"), ((experts + 1,), "int32"),
                 ((padded // 128,), "int32"), ((padded, hidden), "float16")]
        inputs = tuple(Tensor(shape, dtype, f"input{index}_repeat{repeat}")
                       for index, (shape, dtype) in enumerate(specs))
        sets.append(inputs)
        namespace["run_kernel"](*inputs)
        assert len(launches) == 2 * (repeat + 1), (version, experts, routed_dtype)
        assert len(compiled) == 2, "Only compiled kernels, not results, may be cached"
        assert len(allocated) == 1, "Workspace is reused"
        s1, s2 = launches[-2:]
        assert s1[2] == tuple(inputs[index] for index in (0, 1, 2, 5, 7, 8)) + (allocated[0],)
        assert s2[2] == (allocated[0],) + tuple(inputs[index] for index in (3, 4, 5, 6, 7, 8, 9))
        assert s2[1][-1] == routed_dtype
    assert all(first is not second for first, second in zip(*sets))
    return [(name, args, configs) for name, args, configs in compiled]


for experts in (1, 8, 16, 32, 64):
    for dtype in ("float16", "float32"):
        actual = check_mock(716, experts, dtype)
        reference = 715 if experts == 64 else 714
        expected = check_mock(reference, experts, dtype)
        assert actual == expected, (experts, dtype, actual, expected)
        for name, _, _ in actual:
            assert segment(716, name) == segment(reference, name)
        print(f"E{experts:02} {dtype}: same complete path as v{reference}; 2 launches x 2 fresh calls PASS")

print("Complete-source minimal-diff, decorated builder AST/source, dispatch, caching checks PASS")
print("No GPU correctness or performance conclusion is implied by these CPU mock checks")
print("SHA256", hashlib.sha256(FILES[716].read_bytes()).hexdigest())
