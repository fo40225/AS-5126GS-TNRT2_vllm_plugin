"""JIT build and loading of the ``_muse_topo_C`` extension, plus the IPC workspace helpers.

The sources live in ``csrc`` (every ``*.cu`` / ``*.cpp`` there is compiled) and are built with
``torch.utils.cpp_extension.load`` into ``$VLLM_CACHE_ROOT/muse_topo/<digest>``. The directory name hashes the
sources, flags, torch version and nvcc, and the build is guarded by an flock, so concurrent ranks build once.
Importing this module never builds; call :func:`load`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import torch
from torch.utils import cpp_extension

_EXT_NAME = "_muse_topo_C"
_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_CSRC_SUFFIXES = (".cpp", ".cu", ".cuh", ".h")
_MIN_NVCC_VERSION = (12, 8)
_BUILT_MARKER = "built"
_BUILD_FLOCK = "build.flock"
_TORCH_BUILD_LOCK = "lock"

_lock = threading.Lock()
_module = None


def _log(msg: str) -> None:
    if os.environ.get("MUSE_TOPO_VERBOSE", "0") == "1":
        print(f"[muse_topo.ops] {msg}", file=sys.stderr, flush=True)


def _cache_root() -> str:
    root = os.environ.get("VLLM_CACHE_ROOT")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "vllm")
    return os.path.join(root, "muse_topo")


def _nvcc_path() -> str:
    cuda_home = cpp_extension.CUDA_HOME
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else ""
    if not os.path.isfile(nvcc):
        raise RuntimeError("muse_topo: nvcc not found; set CUDA_HOME to a CUDA toolkit (>= 12.8)")
    return nvcc


def _nvcc_version(nvcc: str) -> tuple[int, int]:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True, timeout=60).stdout
    m = re.search(r"release (\d+)\.(\d+)", out)
    if m is None:
        raise RuntimeError(f"cannot parse nvcc version: {out!r}")
    v = (int(m.group(1)), int(m.group(2)))
    if v < _MIN_NVCC_VERSION:
        raise RuntimeError(f"muse_topo needs nvcc >= {_MIN_NVCC_VERSION}, found {v} at {nvcc}")
    return v


def _gencode(v: tuple[int, int]) -> list[str]:
    archs = ["120f"] if v >= (13, 0) else (["120a", "121a"] if v >= (12, 9) else ["120a"])
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]


def _flags(v: tuple[int, int]) -> tuple[list[str], list[str]]:
    # No --use_fast_math: the arithmetic contract needs IEEE fp32 adds, no contraction or flushing.
    cflags = ["-O3"]
    trace = ["-DMUSE_TOPO_TRACE"] if os.environ.get("MUSE_TOPO_TRACE") == "1" else []
    cuda_cflags = ["-O3", *trace, *_gencode(v), "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                   "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "-lineinfo"]
    return cflags, cuda_cflags


def _sources() -> list[str]:
    return sorted(str(p) for p in _CSRC_DIR.iterdir() if p.suffix in (".cpp", ".cu"))


def _build_dir(nvcc: str, cflags: list[str], cuda_cflags: list[str]) -> str:
    digest = hashlib.sha256()
    for p in sorted(_CSRC_DIR.iterdir()):
        if p.suffix in _CSRC_SUFFIXES:
            digest.update(p.name.encode())
            digest.update(p.read_bytes())
    digest.update(repr((cflags, cuda_cflags)).encode())
    digest.update(f"{torch.__version__}|{nvcc}".encode())
    return os.path.join(_cache_root(), digest.hexdigest()[:16])


def _built_library(build_dir: str) -> str | None:
    try:
        name = Path(build_dir, _BUILT_MARKER).read_text().strip()
    except OSError:
        return None
    lib = os.path.join(build_dir, name)
    return lib if name and os.path.isfile(lib) else None


@contextlib.contextmanager
def _build_exclusive(build_dir: str):
    with open(os.path.join(build_dir, _BUILD_FLOCK), "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(f"another process is building {build_dir}; waiting")
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _import_library(path: str):
    spec = importlib.util.spec_from_file_location(_EXT_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load():
    """Return the compiled extension module, building it on first use (flock-guarded)."""
    global _module
    with _lock:
        if _module is not None:
            return _module
        nvcc = _nvcc_path()
        v = _nvcc_version(nvcc)
        cflags, cuda_cflags = _flags(v)
        build_dir = _build_dir(nvcc, cflags, cuda_cflags)
        os.makedirs(build_dir, exist_ok=True)
        lib = _built_library(build_dir)
        if lib is None:
            with _build_exclusive(build_dir):
                lib = _built_library(build_dir)
                if lib is None:
                    try:  # a torch lock file left by a killed builder would block forever
                        os.remove(os.path.join(build_dir, _TORCH_BUILD_LOCK))
                    except FileNotFoundError:
                        pass
                    _log(f"building {_EXT_NAME} (nvcc {v[0]}.{v[1]}) in {build_dir}")
                    # is_python_module=False returns the path of the built library without importing it
                    lib = cpp_extension.load(
                        name=_EXT_NAME, sources=_sources(), extra_include_paths=[str(_CSRC_DIR)],
                        extra_cflags=cflags, extra_cuda_cflags=cuda_cflags, build_directory=build_dir,
                        with_cuda=True, is_python_module=False, verbose=os.environ.get("MUSE_TOPO_VERBOSE") == "1")
                    Path(build_dir, _BUILT_MARKER).write_text(os.path.basename(lib))
        _module = _import_library(lib)
        return _module


# ------------------------------------------------------------------------------------------------
# workspace + IPC
# ------------------------------------------------------------------------------------------------
class Workspace:
    """Per-rank receive buffers, exchanged through CUDA IPC.

    ``handles`` are the 64-byte IPC handles of all ranks (``all_gather_object`` over the gloo group); the local
    rank uses its own pointer, the others are opened with lazy peer access.
    """

    def __init__(self, ext, device: torch.device, nbytes: int):
        self.ext = ext
        self.device = device
        self.nbytes = nbytes
        with torch.cuda.device(device):
            self.ptr, self.handle = ext.ws_alloc(nbytes)
        self.peer_ptrs: dict[int, int] = {}
        self.bases: list[int] = []

    def exchange(self, cpu_group, rank: int, world_size: int) -> list[int]:
        import torch.distributed as dist

        handles = [None] * world_size
        dist.all_gather_object(handles, self.handle, group=cpu_group)
        bases = []
        with torch.cuda.device(self.device):
            for r in range(world_size):
                if r == rank:
                    bases.append(self.ptr)
                else:
                    p = self.ext.ws_open(handles[r])
                    self.peer_ptrs[r] = p
                    bases.append(p)
        self.bases = bases
        return bases

    def close(self) -> None:
        with torch.cuda.device(self.device):
            for p in self.peer_ptrs.values():
                self.ext.ws_close_peer(p)
            self.peer_ptrs.clear()
            if self.ptr:
                self.ext.ws_free(self.ptr)
                self.ptr = 0
