"""CPU-only unit test of the vllm_plugin wrappers with a mock TopoComm (no GPU, no vLLM objects).

usage: python bench/test_plugin_unit.py        (needs torch and the host folder on PYTHONPATH; takes a second)
Checks the routing logic: topo path only when supports() holds and the op is routed, dim normalization, the dim != 0
reduce_scatter fallback, the lazy fallback reason, the per-key counters, MUSE_TOPO_OPS parsing, and that register() is a
no-op without MUSE_TOPO_ENABLE=1, and the RS + AG == AR capability.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from muse_topo import vllm_plugin as vp  # noqa: E402

# the trace asks the CUDA runtime for the current stream; the tensors here live on the CPU
torch.cuda.current_stream = lambda dev=None: types.SimpleNamespace(cuda_stream=0x1234)
torch.cuda.is_current_stream_capturing = lambda: False


class FakeComm:
    device = torch.device("cpu")
    max_bytes = 8 << 20
    world_size = 8
    _closed = False

    def __init__(self, ok):
        self.ok, self.calls = ok, []

    def supports(self, op, t, dim=None):
        self.calls.append(("supports", op, dim))
        return self.ok

    # with ``out`` (a result pre-allocated by the plugin, see _output_stream) the fake reports the shape it was given
    def all_reduce(self, x, out=None):
        return "topo_ar" if out is None else ("topo_ar", tuple(out.shape))

    def all_gather(self, x, dim, out=None):
        return ("topo_ag", dim) if out is None else ("topo_ag", dim, tuple(out.shape))

    def reduce_scatter(self, x, out=None):
        return "topo_rs" if out is None else ("topo_rs", tuple(out.shape))


def make(ok, ops=("all_reduce", "all_gather", "reduce_scatter")):
    dc = types.SimpleNamespace(topo_comm=FakeComm(ok), topo_trace=vp._OpTrace(0), topo_ops=frozenset(ops))
    return dc


def main():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    ar, ag, rs = (vp._wrap_all_reduce(lambda self, t: "orig_ar"), vp._wrap_all_gather(lambda self, t, d=-1: ("orig_ag", d)),
                  vp._wrap_reduce_scatter(lambda self, t, d=-1: ("orig_rs", d)))
    dc = make(True)
    assert ar(dc, x) == "topo_ar"
    assert ag(dc, x, -1) == ("topo_ag", 1) and ag(dc, x, 0) == ("topo_ag", 0)  # negative dim normalized
    assert rs(dc, x, 0) == "topo_rs" and rs(dc, x, -2) == "topo_rs"
    assert rs(dc, x, 1) == ("orig_rs", 1) and rs(dc, x) == ("orig_rs", -1)  # dim != 0 (default -1 = dim 1): NCCL path
    dc = make(False)
    assert ar(dc, x) == "orig_ar" and ag(dc, x, 0) == ("orig_ag", 0) and rs(dc, x, 0) == ("orig_rs", 0)
    for _ in range(5):
        ar(dc, x)
    assert dc.topo_trace.counts[("all_reduce", None, (4, 64), torch.bfloat16, 0x1234, False, "nccl")] == 6
    # reasons are evaluated lazily: only once per key
    n = []
    dc.topo_trace.note("all_reduce", x, None, "nccl", lambda: n.append(1) or "r")
    dc.topo_trace.note("all_reduce", x, None, "nccl", lambda: n.append(1) or "r")
    assert len(n) == 0  # this key was already seen above
    y = torch.zeros(2, 64, dtype=torch.bfloat16)
    dc.topo_trace.note("all_reduce", y, None, "nccl", lambda: n.append(1) or "r")
    dc.topo_trace.note("all_reduce", y, None, "nccl", lambda: n.append(1) or "r")
    assert len(n) == 1
    # inside a microbatched prefill the result is pre-allocated on the compute stream (shape = what TopoComm would make)
    streams = []
    real_output_stream, real_alloc = vp._output_stream, vp._alloc
    vp._output_stream = lambda: "compute_stream"
    vp._alloc = lambda stream, shape, like: streams.append(stream) or torch.empty(tuple(shape), dtype=like.dtype)
    try:
        dc = make(True)
        assert ar(dc, x) == ("topo_ar", (4, 64))
        assert ag(dc, x, -1) == ("topo_ag", 1, (4, 512))  # column concatenation
        assert ag(dc, x, 0) == ("topo_ag", 0, (32, 64))  # rank-major
        assert ag(dc, torch.zeros(16, dtype=torch.bfloat16), 0) == ("topo_ag", 0, (128,))  # 1-D: rank-major
        assert ag(dc, torch.zeros(4, 6, 8, dtype=torch.bfloat16), -1) == ("topo_ag", 2, (4, 6, 64))
        assert rs(dc, torch.zeros(16, 64, dtype=torch.bfloat16), 0) == ("topo_rs", (2, 64))
        assert streams == ["compute_stream"] * 6
        dc = make(False)  # a call the topo path cannot take allocates nothing
        streams.clear()
        assert ar(dc, x) == "orig_ar" and not streams
    finally:
        vp._output_stream, vp._alloc = real_output_stream, real_alloc
    assert vp._output_stream() is None  # outside a microbatch thread there is no allocation stream (vLLM or not)
    # an op that is not routed never reaches supports()
    dc = make(True, ops=("all_gather",))
    assert ar(dc, x) == "orig_ar" and not any(c[1] == "all_reduce" for c in dc.topo_comm.calls)
    # no communicator (non-TP group, TopoUnavailable): pass through
    dc = types.SimpleNamespace(topo_comm=None, topo_trace=None, topo_ops=frozenset())
    assert ar(dc, x) == "orig_ar" and ag(dc, x, 0) == ("orig_ag", 0) and rs(dc, x, 0) == ("orig_rs", 0)
    dc = types.SimpleNamespace()  # an object that never went through the patched __init__
    assert ar(dc, x) == "orig_ar"
    # why-not reasons (the size / dtype branches need CUDA tensors; they are exercised by test_plugin.py)
    c = FakeComm(False)
    c.large, c.rs_min_bytes = True, 128 << 10
    assert "device" in vp._why_not(c, "all_reduce", torch.zeros(2048, 8192, dtype=torch.bfloat16), None)

    class FakeCuda:  # just enough of a tensor for the size / dtype branches
        is_cuda, device = True, c.device

        def __init__(self, shape, dtype=torch.bfloat16, esize=2):
            self.shape, self.dtype, self._es = shape, dtype, esize

        def numel(self):
            n = 1
            for d in self.shape:
                n *= d
            return n

        def element_size(self):
            return self._es

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 0

    assert "dtype" in vp._why_not(c, "all_reduce", FakeCuda((4, 4096), torch.int32, 4), None)
    assert "multiple of 64" in vp._why_not(c, "all_reduce", FakeCuda((1, 100)), None)
    assert "multiple of 128" in vp._why_not(c, "all_reduce", FakeCuda((1, (8 << 20) // 2 + 32)), None)  # 8 MiB + 64 B
    assert "> 2147483648" in vp._why_not(c, "all_reduce", FakeCuda((1 << 20, (1 << 10) + 64)), None)  # above 2 GiB
    c.large = False
    assert "MUSE_TOPO_LARGE=0" in vp._why_not(c, "all_reduce", FakeCuda((2048, 8192)), None)
    c.large = True
    assert "multiple of 8" in vp._why_not(c, "reduce_scatter", FakeCuda((12, 8192)), 0)
    assert "minimum" in vp._why_not(c, "reduce_scatter", FakeCuda((8, 512)), 0)
    assert "multiple of 128" in vp._why_not(c, "reduce_scatter", FakeCuda((8, 8196)), 0)
    c._closed = True
    assert "closed" in vp._why_not(c, "all_reduce", x, None)
    # _attach never lets an exception of the TopoComm construction escape (vLLM keeps its own collectives)
    import muse_topo

    real = muse_topo.TopoComm
    try:
        for exc in (RuntimeError("boom"), ValueError("bad sysfs"), muse_topo.TopoUnavailable("no")):
            def boom(*a, _e=exc, **k):
                raise _e

            muse_topo.TopoComm = boom
            dc = types.SimpleNamespace(unique_name="tp:0", world_size=8, rank=0, cpu_group=None, device="cpu")
            vp._attach(dc, frozenset(vp._ALL_OPS))
            assert dc.topo_comm is None and dc.topo_ops == frozenset(), exc
        muse_topo.TopoComm = lambda *a, **k: FakeComm(True)
        dc = types.SimpleNamespace(unique_name="tp:0", world_size=8, rank=0, cpu_group=None, device="cpu")
        vp._attach(dc, frozenset(vp._ALL_OPS))
        assert dc.topo_comm is not None
        dc = types.SimpleNamespace(unique_name="ep:0", world_size=8, rank=0, cpu_group=None, device="cpu")
        vp._attach(dc, frozenset(vp._ALL_OPS))
        assert dc.topo_comm is None  # not the TP group
    finally:
        muse_topo.TopoComm = real
    # capability reduce_scatter_all_gather_matches_all_reduce: needs a TopoComm, all three ops routed, the large
    # algorithms and the exact fp32 exchange
    Cap = type("Cap", (), {"reduce_scatter_all_gather_matches_all_reduce": property(vp._rs_ag_matches_ar)})
    full = frozenset(vp._ALL_OPS)

    def cap(comm, ops=full):
        obj = Cap()
        obj.topo_comm, obj.topo_ops = comm, ops
        return obj.reduce_scatter_all_gather_matches_all_reduce

    c = FakeComm(True)
    c.large, c.bf16_exchange = True, False
    assert cap(c) is True
    assert cap(None, frozenset()) is False  # no TopoComm (not the TP group, host mismatch, MUSE_TOPO_ENABLE=0)
    for missing in vp._ALL_OPS:
        assert cap(c, full - {missing}) is False, missing
    c.large = False
    assert cap(c) is False  # reduce_scatter and the large all-reduce would go to NCCL
    c.large, c.bf16_exchange = True, True
    assert cap(c) is False  # the large algorithms round the partials, the small all-reduce does not
    assert not Cap().__class__.__dict__["reduce_scatter_all_gather_matches_all_reduce"].fget(types.SimpleNamespace())
    # fused collective + epilogue methods (the optional interface of the model's ops/fused_boundary.py)
    class FusedFake(FakeComm):
        def __init__(self, ok, params_ok=True):
            super().__init__(ok)
            self.params_ok = params_ok

        def supports_boundary_rows(self, kind, n):
            return self.ok and 1 <= n <= 64

        def supports_all_reduce_boundary(self, x, stream):
            return self.ok

        def supports_all_gather_boundary(self, shard, stream):
            return self.ok

        def supports_all_gather_rmsnorm(self, shard):
            return self.ok

        def boundary_params_ok(self, *a, **k):
            return self.params_ok

        def _boundary_ok(self, t, shape, dtype):
            return self.params_ok

        def all_reduce_boundary(self, *a, **k):
            return ("topo_arb", k.get("out_fp32"))

        def all_gather_boundary(self, *a, **k):
            return ("topo_agb", k.get("out_fp32"))

        def all_gather_rmsnorm(self, *a, **k):
            return "topo_agr"

    kw = dict(alpha=x, beta=x, pre_weight=x, post_weight=None, post_eps=1e-8, pre_eps=1e-6, post_gain_center=0.0,
              pre_gain_center=1.0)

    def fused_dc(ok=True, params_ok=True, ops=("all_reduce", "all_gather", "reduce_scatter", "boundary")):
        return types.SimpleNamespace(topo_comm=FusedFake(ok, params_ok), topo_trace=vp._OpTrace(0),
                                     topo_ops=frozenset(ops))

    dc = fused_dc()
    assert vp._supports_fused_residual_boundary(dc, "all_reduce_boundary", 8)
    assert not vp._supports_fused_residual_boundary(dc, "all_reduce_boundary", 65)
    assert vp._all_reduce_residual_boundary(dc, x, x, **kw) == ("topo_arb", None)
    assert vp._all_gather_residual_boundary(dc, x, x, out_fp32=x, **kw) == ("topo_agb", x)
    assert vp._all_gather_rmsnorm(dc, x, x, 1e-6, 1.0) == "topo_agr"
    assert dc.topo_trace.counts[("all_reduce_boundary", None, (4, 64), torch.bfloat16, 0x1234, False, "topo")] == 1
    for dc in (fused_dc(ok=False), fused_dc(params_ok=False), fused_dc(ops=("all_reduce", "all_gather")),
               types.SimpleNamespace(topo_comm=None, topo_trace=None, topo_ops=frozenset()), types.SimpleNamespace()):
        assert vp._all_reduce_residual_boundary(dc, x, x, **kw) is None
        assert vp._all_gather_residual_boundary(dc, x, x, **kw) is None
        assert vp._all_gather_rmsnorm(dc, x, x, 1e-6, 1.0) is None
    assert not vp._supports_fused_residual_boundary(fused_dc(ops=("all_reduce",)), "all_gather_rmsnorm", 1)
    assert not vp._supports_fused_residual_boundary(types.SimpleNamespace(), "all_gather_rmsnorm", 1)
    assert "boundary" in vp._ALL_OPS
    # MUSE_TOPO_OPS parsing and the register() gate
    os.environ["MUSE_TOPO_OPS"] = "all_reduce, bogus"
    assert vp._routed_ops() == frozenset({"all_reduce"})
    del os.environ["MUSE_TOPO_OPS"]
    assert vp._routed_ops() == frozenset(vp._ALL_OPS)
    os.environ.pop("MUSE_TOPO_ENABLE", None)
    vp.register()  # no-op, must not import vLLM
    print("test_plugin_unit PASSED")


if __name__ == "__main__":
    main()
