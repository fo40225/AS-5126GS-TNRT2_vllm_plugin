"""Build (or load) the _muse_topo_C extension outside the GPU lock. usage: py_topo.sh bench/build_ext.py"""
import time

from muse_topo import ops

t = time.time()
m = ops.load()
print("loaded", m, f"{time.time() - t:.1f}s")
print(m.layout(8 << 20, 256 << 10, 64 << 10, 1 << 20))
