"""
Debug bootstrap: runs the given file as part of its Python package so that
relative imports work. Equivalent to `python -m <module>` but driven by file path.

Usage (via launch.json):
    program: run_as_module.py
    args:    ["${file}"]
"""
import runpy
import sys
import os

if len(sys.argv) < 2:
    sys.exit("Usage: run_as_module.py <file_path>")

file_path = os.path.abspath(sys.argv[1])
parts = file_path.split(os.sep)

# Walk up the path to find the 'src' directory whose immediate child is a package.
src_idx = next(
    (
        i
        for i in range(len(parts) - 2, 0, -1)
        if parts[i] == "src"
        and os.path.isfile(
            os.path.join(os.sep.join(parts[: i + 2]), "__init__.py")
        )
    ),
    None,
)

if src_idx is None:
    sys.exit(f"Could not locate a 'src' package root for: {file_path}")

src_root = os.sep.join(parts[: src_idx + 1])
if src_root not in sys.path:
    sys.path.insert(0, src_root)

module_name = ".".join(parts[src_idx + 1 :])
if module_name.endswith(".py"):
    module_name = module_name[:-3]

runpy.run_module(module_name, run_name="__main__", alter_sys=True)
