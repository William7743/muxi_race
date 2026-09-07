"""Export compiler-generated source for inspection, never a submission runner."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('inspected_kernel', a.source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for dim in (32, 64):
        kernel = mod._get_kernel(4, 1024, 1, 16, dim, 1, 16, 1)
        code = kernel.get_kernel_source()
        target = a.output_dir / f'd{dim}_bs16.txt'
        target.write_text(code)
        print(json.dumps({'dim': dim, 'source_chars': len(code),
                          'output': target.name}), flush=True)


if __name__ == '__main__':
    main()
