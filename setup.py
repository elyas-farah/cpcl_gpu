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
    import shutil
    import subprocess
    from ctypes.util import find_library

    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

    # Probe for BLAS: pkg-config (gives include path too), then ldconfig cache.
    _blas_found = False
    if shutil.which("pkg-config"):
        for _pkg in ("openblas", "blas", "cblas"):
            try:
                subprocess.check_call(
                    ["pkg-config", "--exists", _pkg], stderr=subprocess.DEVNULL
                )
                for _f in subprocess.check_output(
                    ["pkg-config", "--cflags", _pkg], stderr=subprocess.DEVNULL
                ).decode().split():
                    if _f.startswith("-I"):
                        include_dirs.append(_f[2:])
                for _f in subprocess.check_output(
                    ["pkg-config", "--libs", _pkg], stderr=subprocess.DEVNULL
                ).decode().split():
                    if _f.startswith("-L"):
                        library_dirs.append(_f[2:])
                    elif _f.startswith("-l"):
                        libraries.append(_f[2:])
                _blas_found = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

    if not _blas_found:
        for _name in ("openblas", "blas"):
            if find_library(_name):
                libraries.append(_name)
                _blas_found = True
                break

    if not _blas_found:
        # Last resort — user may need to install libopenblas-dev or libblas-dev.
        libraries.append("openblas")

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
