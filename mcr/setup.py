from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent

if Path.cwd().resolve() != ROOT:
    os.chdir(ROOT)


extension = Extension(
    name="mcr.avinfra_persuasion.routing._mosp_ext",
    sources=[
        "src/mcr/avinfra_persuasion/routing/_mosp_ext.pyx",
        "src/mcr/avinfra_persuasion/routing/mosp_adapter.cpp",
        "src/mcr/avinfra_persuasion/routing/mod/graph/src/graph.cpp",
        "src/mcr/avinfra_persuasion/routing/mod/preprocessing/src/Preprocessor.cpp",
        "src/mcr/avinfra_persuasion/routing/mod/search/src/SolutionsList.cpp",
    ],
    include_dirs=[
        "src/mcr/avinfra_persuasion",
        "src/mcr/avinfra_persuasion/routing",
        "src/mcr/avinfra_persuasion/routing/mod/datastructures/includes",
        "src/mcr/avinfra_persuasion/routing/mod/graph/includes",
        "src/mcr/avinfra_persuasion/routing/mod/preprocessing/includes",
        "src/mcr/avinfra_persuasion/routing/mod/search/includes",
        np.get_include(),
    ],
    define_macros=[
        ("MCR_MOSP_DIM", "7"),
    ],
    extra_compile_args=[
        "-std=c++20",
    ],
    language="c++",
)


setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": "3"},
    ),
)
