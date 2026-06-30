import sys
import sysconfig
from pathlib import Path

import pybind11
from setuptools import Extension, setup

extra_compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden"]
extra_link_args = []
include_dirs = [pybind11.get_include()]
library_dirs = []
libraries = []
define_macros = []

if sys.platform == "darwin":
    # Apple clang needs an explicit OpenMP runtime (libomp from Homebrew);
    # Accelerate provides a BLAS (cblas) implementation for the block GEMMs.
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp", "-framework", "Accelerate"]

    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        p = Path(prefix)
        if p.exists():
            include_dirs.append(str(p / "include"))
            library_dirs.append(str(p / "lib"))
            break
else:
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]
    libraries += ["openblas"]

ext_modules = [
    Extension(
        "cpcl_cov",
        sources=["src/get_covariance.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        language="c++",
    ),
]

setup(
    name="cpcl_cov",
    version="0.1.0",
    description="OpenMP-parallel C++ pseudo-Cl brute-force covariance kernel",
    ext_modules=ext_modules,
)
