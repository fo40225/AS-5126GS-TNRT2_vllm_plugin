"""TopoComm: P2P collectives specialized for the 8-GPU PCIe box (2 sockets x 4 GPUs, see topology.py).

Public API (stable; the vLLM plugin and the large-message algorithms build on it)::

    comm = TopoComm(cpu_group, device, rank, world_size, *, max_bytes=8 << 20, check_env=True)
    comm.supports(op, tensor, dim=None) -> bool   # op in {"all_reduce", "all_gather", "reduce_scatter"}
    comm.all_reduce(x, out=None)       -> tensor, same shape/dtype as x (new unless ``out`` is given)
    comm.all_gather(x, dim, out=None)  -> new tensor; dim=-1 (last): out[..., o*w:(o+1)*w] = x_o (column
                                          concatenation), dim=0: cat([x_0 .. x_7], 0) (rank-major, NCCL layout)
    comm.reduce_scatter(x, out=None)   -> rows block r of the reduced [T, ...] tensor (T % 8 == 0), r = my rank
    comm.close()

Construction is collective (every rank of the TP group must call it with the same ``max_bytes``): it derives the
topology (``TopoUnavailable`` if the host does not match or ``MUSE_TOPO_ENABLE != 1`` and ``check_env``),
allocates the IPC workspace once (about 59 MiB per GPU for ``max_bytes=8 MiB``) and exchanges the handles over
``cpu_group`` (a gloo group).

``supports`` is the only eligibility check; callers fall back to NCCL when it returns False. It depends on the call's
dtype/shape/contiguity/device only, never on rank-local state such as the data pointer, so every rank of a collective
takes the same path (a topo rank waiting for an NCCL rank would deadlock). Operands whose data pointer is not 16-byte
aligned (views with odd offsets) are staged through an aligned copy inside the ops. Conditions: CUDA tensor on
the communicator's device, contiguous; all_reduce: bf16/fp16/fp32 and a size that is a
multiple of 64 B and at most ``max_bytes`` (decode/medium algorithms), or a multiple of 128 B of any size up to 2 GiB
(large algorithm, chunked reduce-scatter + all-gather through the same workspace); all_gather: any dtype, dim = last or
0 (any other dim: ValueError), shard bytes a multiple of 4, and for dim = last the row bytes too (16 for shards
above 64 KiB), shards of any size (above 1 MiB: chunked large algorithm, 16 B multiples); reduce_scatter: bf16/fp16/fp32, dim-0 row blocks, T % 8 == 0, block bytes a multiple
of 16 and at least ``rs_min_bytes`` of message. The large algorithms and the size thresholds are described below.

Large messages (prefill): the message is 8 equal blocks (block r = rows block r of a [T, H] tensor), processed in chunks
of ``large_chunk_bytes`` (1 MiB) per block. reduce_scatter(chunk) leaves the reduced chunk of block r on rank r (the
fp32 partial of the partner's block crosses the socket cut once: S/4 bytes per GPU); all_gather(chunk) distributes it.
all_reduce = both, fused into one pipelined launch per chunk. Bitwise identical to the small algorithms, and
all_gather(reduce_scatter(x), dim 0) == all_reduce(x) bitwise. ``out`` of all_reduce may alias ``x``.
Opt-in ``MUSE_TOPO_LARGE_BF16_EXCHANGE=1`` (or ``bf16_exchange=True``): the two island partials are rounded to bf16
before the final add (half the cut bytes, class deterministic_different, still rank-identical and run-to-run
deterministic). It applies to bf16 all_reduce/reduce_scatter calls served by the large algorithm and, unless
``MUSE_TOPO_MEDIUM_BX=0``, to the bf16 all_reduce calls of the medium algorithm (algorithm 2, ``BX`` kernel variant)
from ``ar_bx_min`` (64 KiB) up to ``max_bytes``, with the same arithmetic as the large algorithm, so every bf16
all_reduce of at least 64 KiB gets the same bits whichever algorithm its size selects (docs/r3c_medium_ar.md); fp16
and fp32 keep the exact exchange (an fp16 partial above 65504 would overflow) and the flag and the medium-exchange
settings (``MUSE_TOPO_MEDIUM_BX``, ``MUSE_TOPO_MEDIUM_BX_MIN_BYTES``, the tuning file) must be the same on all ranks.
reduce_scatter always runs the large algorithm with the exchange when the flag is on (it needs at least
``rs_min_bytes``, 128 KiB), so reduce_scatter and all_reduce agree bitwise from 128 KiB on (from 2 MiB with
``MUSE_TOPO_MEDIUM_BX=0``). Messages below ``ar_bx_min`` and the fused all-reduce + boundary kernels (T <= 8) keep the
exact fp32 exchange, so an element can still get different bits depending on its message size: at ``ar_bx_min``
(64 KiB; 2 MiB with ``MUSE_TOPO_MEDIUM_BX=0``), which in decode is the embedding all-reduce [T, 8192] at T 3 -> 4 and
the MoE-out all-reduce [T, 4096] at T 7 -> 8 (eager; padded T 2 -> 4 and 4 -> 8 in CUDA graphs), and, with the medium
exchange on, between the fused and the unfused T = 4 .. 8 attn-out reduction.
``MUSE_TOPO_LARGE=0`` disables the large algorithms (supports() then keeps the H1a limits).

Calls are asynchronous on the current CUDA stream, CUDA-graph safe (no host state per call), and must be issued
in the same order with the same shapes on all ranks.

Stream rule: the double-buffered receive slots and the per-GPU call counter are only safe if the collectives of one
communicator execute in stream order, one at a time. Eager calls therefore must stay on ONE stream. If a call
arrives on another stream, the new stream first waits for everything enqueued so far on the previous stream (an
event is recorded on it), which keeps the collectives totally ordered; with MUSE_TOPO_STRICT_STREAM=1 it raises
instead. Calls issued while a CUDA graph is being captured are not checked (the kernels are replayed later): graph
replays of one communicator must be launched on the same stream as the eager calls (vLLM: the main stream).

Arithmetic contract of all_reduce (every size and algorithm; bitwise identical on all ranks and runs)::

    P_I = ((x_{m0} + x_{m1}) + x_{m2}) + x_{m3}   in fp32, m0..m3 = ranks of island I ascending
    out = round_to_dtype(P_0 + P_1)               in fp32, single final rounding

With the bf16 exchange (bf16 inputs, see above) the contract is ``out = round_bf16(bf16(P_0) + bf16(P_1))``: each island
partial is rounded to bf16 (round to nearest even), the sum of the two is formed in fp32 and rounded once.

Islands are the two sockets ({0,1,2,3} and {4,5,6,7} on this host); partner(r) = r ^ 4. Words of the exchanged
payload that equal 0xFFFFFFFF (a NaN pattern) are replaced by 0xFFFFFFFE, so NaN payloads may change (NaN stays NaN).
all_gather is bit exact for any payload.
"""

from __future__ import annotations

import logging
import os

import torch

from . import ops
from .topology import TopoInfo, TopoUnavailable, derive, env_enabled, local_descriptor

logger = logging.getLogger("vllm.muse_topo")

_AR_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_OPS = ("all_reduce", "all_gather", "reduce_scatter")
MAX_GRID = 128
WORLD = 8

# Size dispatch tables: (largest nbytes covered, algorithm, grid, block threads, unroll).
# Measured on the AS-5126GS-TNRT2 box on 2026-10-23 with bench/bench_h1a.py --sweep (8 processes, CUDA graph of
# back-to-back calls, bf16, NCCL_P2P_LEVEL=SYS baseline): the winning configuration of every measured size is applied
# to the bucket that ends at that size (bench/make_table.py generates these literals; sizes up to 512 KiB (all_reduce)
# and 64 KiB (all_gather) come from the final kernels' sweep, larger sizes from the sweep before the store-on-arrival
# change, whose best configurations are equivalent). Algorithm ids: all_reduce 1 = 2-phase sentinel, 2 = 3-phase
# sentinel, 3 = 3-phase flag; all_gather 1 = LL8, 2 = flag. Algorithms 1 and 3 of all_reduce never won at any size
# (they stay available through TopoComm._all_reduce for tests and experiments).
# NOTE (H1b): rows from 2 MiB on are only reached by explicit _all_reduce(...) calls (tests, benchmarks): the default dispatch sends messages
# >= _AR_LARGE_MIN (2 MiB) to _AR_LARGE_TABLE below.
_AR_TABLE = [
    (8192, 2, 4, 32, 1),
    (16384, 2, 4, 64, 1),
    (32768, 2, 8, 64, 1),
    (65536, 2, 32, 32, 1),
    (131072, 2, 64, 32, 1),
    (262144, 2, 64, 32, 1),
    (524288, 2, 8, 256, 1),
    (1048576, 2, 8, 256, 1),
    (2097152, 2, 8, 256, 1),
    (4194304, 2, 8, 256, 1),
    (1 << 62, 2, 4, 256, 2),
]
# all_reduce with the bf16 exchange of the island partials (algorithm 2 only, bf16 inputs, used from _AR_BX_MIN bytes when the bf16
# exchange is on); same row format. Measured with bench/bench_medium.py on 2026-10-26 (8 processes, CUDA graph of back-to-back calls,
# bf16 [T, 8192]; sweep of grid x block x unroll, every candidate verified bitwise against the torch emulation, best configuration
# retimed with the full protocol, fewest threads within 1.5 %); the comments give the b2b time and, in brackets, the exact-exchange
# time of the best _AR_TABLE-style configuration at the same size. docs/r3c_medium_ar.md has the full table.
_AR_BX_TABLE = [
    (65536, 2, 8, 128, 1),  # 64 KiB 6.7 us (exact 8.1)
    (98304, 2, 6, 256, 1),  # 96 KiB 7.7 (9.9)
    (131072, 2, 8, 256, 1),  # 128 KiB 9.0 (11.9)
    (196608, 2, 6, 512, 1),  # 192 KiB 11.5 (16.1)
    (262144, 2, 64, 64, 1),  # 256 KiB 13.8 (18.9)
    (393216, 2, 12, 256, 1),  # 384 KiB 18.6 (25.8)
    (524288, 2, 16, 256, 1),  # 512 KiB 22.5 (33.3)
    (786432, 2, 8, 512, 1),  # 768 KiB 32.8 (47.7)
    (1048576, 2, 8, 512, 1),  # 1 MiB 43.1 (62.5)
    (1572864, 2, 16, 128, 1),  # 1.5 MiB 63.3 (94.6)
    (2097152, 2, 16, 128, 1),  # 2 MiB 82.8 (123.7)
    (3145728, 2, 8, 256, 1),  # 3 MiB 122.2 (183.0)
    (4194304, 2, 16, 128, 1),  # 4 MiB 161.4 (243.8)
    (6291456, 2, 16, 128, 1),  # 6 MiB 241.0 (357.9)
    (8388608, 2, 16, 128, 1),  # 8 MiB 321.3 (480.3)
    (1 << 62, 2, 16, 128, 1),
]
_AG_TABLES = {
    0: [  # layout 0: dim = -1 (column concatenation)
        (1024, 1, 2, 128, 1),
        (2048, 1, 4, 128, 1),
        (4096, 1, 8, 128, 1),
        (8192, 1, 64, 32, 1),
        (16384, 1, 128, 32, 1),
        (32768, 2, 32, 128, 1),
        (50512, 2, 24, 256, 1),
        (65536, 2, 16, 256, 1),
        (131072, 2, 32, 1024, 1),
        (262144, 2, 32, 1024, 1),
        (1 << 62, 2, 32, 256, 1),
    ],
    1: [  # layout 1: dim = 0 (rank-major)
        (1024, 1, 2, 128, 1),
        (2048, 1, 4, 128, 1),
        (4096, 1, 8, 128, 1),
        (8192, 1, 64, 32, 1),
        (16384, 1, 128, 32, 1),
        (32768, 2, 16, 256, 1),
        (50512, 2, 24, 256, 1),
        (65536, 2, 16, 256, 1),
        (131072, 2, 32, 1024, 1),
        (262144, 2, 32, 1024, 1),
        (1 << 62, 2, 32, 256, 1),
    ],
}


# Large-message configurations (bench/bench_h1b.py --search, coordinate descent per size, verified bitwise; AS-5126GS-TNRT2,
# 8 processes, CUDA graph of back-to-back calls, bf16 [T, 8192], NCCL_P2P_LEVEL=SYS baseline; 2026-10-23, details and the
# measured tables in docs/h1b_latency.md). Rows end at the largest size they were measured at; larger sizes use the last.
# all_reduce: (largest nbytes covered, mode, chunk bytes, reduce-scatter blocks, all-gather blocks, block threads,
# unroll, cut_rs, cut_ag); mode 1 = fused pipeline step, 0 = two launches per chunk; cut_x = 1: the socket-cut pushes of
# the all-gather role come from one dedicated block (cut_rs = 1 / cut_ag = 2 variants were never faster: the cut is
# shared by both roles and a single staged block cannot feed it better than the workers).
_AR_LARGE_TABLE = [
    (2 << 20, 1, 1 << 20, 4, 32, 512, 1, 0, 1),  # 2 MiB: 117 us (H1a 121, NCCL 134)
    (3 << 20, 0, 1 << 20, 4, 32, 256, 1, 0, 1),  # 2.5 MiB 143 us, 3 MiB 170 us (H1a 182, NCCL 218)
    (6 << 20, 0, 1 << 20, 4, 32, 256, 1, 0, 1),  # 4 MiB 221 us, 6 MiB 329 us (H1a 241 / 356)
    (8 << 20, 1, 1 << 20, 4, 32, 512, 1, 0, 1),  # 8 MiB 449 us (H1a 477, NCCL 579)
    (1 << 62, 1, 1 << 20, 4, 8, 512, 2, 0, 1),  # 16 MiB 906 us, 64 MiB 3561 us, 128 MiB 7130 us (NCCL 1158 / 4611 / 9177)
]
# reduce_scatter: (largest nbytes of the message, chunk bytes, blocks, block threads, unroll, cut)
_RS_TABLE = [
    (1 << 20, 1 << 20, 8, 256, 1, 0),  # 128 KiB 7.7 us (NCCL 27.7), 512 KiB 20 us (38), 1 MiB 36.6 us (49)
    (4 << 20, 1 << 20, 4, 256, 1, 0),  # 2 MiB 66 us (73), 4 MiB 126 us (150)
    (1 << 62, 1 << 20, 8, 256, 1, 0),  # 16 MiB 496 us (598), 64 MiB 1973 us (2373), 128 MiB 4024 us (4790)
]
# large all_gather: (largest shard bytes, chunk bytes, blocks, block threads, cut); used from _AG_LARGE_MIN bytes of shard
_AGL_TABLE = [
    (1 << 62, 1 << 19, 8, 1024, 1),  # out 2 MiB 52 us (H1a 56, NCCL 72), 16 MiB 416 us (479 / 588), 128 MiB 3315 us (3890 / 4660)
]
# Thresholds re-checked in H2b against the raw bench_h1b rows (b2b us, H1a table vs the searched large configuration): 1 MiB 61.3 vs 63.6, 1.5 MiB 91.2 vs
# 90.2 (a tie), 2 MiB 120.4 vs 115.8, 2.5 MiB 150.9 vs 143.0, 3 MiB 179.3 vs 169.2: the crossover is at 2 MiB. The full-model EP8 runs
# (docs/h2_vllm_plugin.md) confirm the tables end to end.
_AR_LARGE_MIN = 2 << 20  # messages below use the H1a algorithms (up to max_bytes): they win at 1 MiB (61.3 vs 63.6 us)
_AR_BX_MIN = 64 << 10  # bf16 all-reduces from this size on use the bf16 exchange of algorithm 2 (when it is on): 6.7 vs 8.1 us at 64 KiB
# With the bf16 exchange algorithm 2 beats the large algorithm at every size up to its workspace limit (1 MiB 43.1 vs 64.3 us,
# 2 MiB 82.8 vs 97.4, 8 MiB 321 vs 365: the large one moves the all-gather half of the result over the cut as well), so the large
# algorithm only takes the messages above max_bytes.
_AR_LARGE_MIN_BX = 1 << 62
_RS_MIN_BYTES = 128 << 10
_AG_LARGE_MIN = 256 << 10  # shards from this size on use the large engine instead of the H1a flag path (10.4 vs 11.4 us at 128 KiB)
_LARGE_MAX_BYTES = 2 << 30


def _ws_align(v: int) -> int:
    return (v + 255) & ~255


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _ag_layout(x: torch.Tensor, dim: int):
    """(layout, row bytes) of an all_gather of ``x`` along ``dim``, or None if the topo path does not gather that dim.

    layout 0 = column concatenation (dim = last dim of a >= 2-D tensor), layout 1 = rank-major (dim 0 and every 1-D
    tensor, NCCL layout). The row of layout 1 is the whole shard: the kernels only use the row of layout 0.
    """
    nd = x.dim()
    if nd == 0:
        return None
    d = dim + nd if dim < 0 else dim
    if d < 0 or d >= nd:
        return None
    if nd == 1 or d == 0:
        return 1, x.numel() * x.element_size()
    if d == nd - 1:
        return 0, x.shape[-1] * x.element_size()
    return None


def _pick(table, nbytes):
    for row in table:
        if nbytes <= row[0]:
            return row
    return table[-1]


class TopoComm:
    __doc__ = __doc__

    def __init__(self, cpu_group, device, rank: int, world_size: int, *, max_bytes: int = 8 << 20,
                 ar1_max_bytes: int = 256 << 10, ag_ll_max_bytes: int = 64 << 10, ag_flag_max_bytes: int = 1 << 20,
                 check_env: bool = True, timeout_s: float | None = None, large_chunk_bytes: int | None = None,
                 bf16_exchange: bool | None = None):
        if timeout_s is None:
            timeout_s = float(os.environ.get("MUSE_TOPO_TIMEOUT_S", "10"))
        if check_env and not env_enabled():
            raise TopoUnavailable("MUSE_TOPO_ENABLE != 1")
        import torch.distributed as dist

        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.max_bytes, self.ar1_max_bytes, self.ag_ll_max_bytes = max_bytes, ar1_max_bytes, ag_ll_max_bytes
        self.ag_flag_max_bytes = ag_flag_max_bytes
        # The large algorithms reuse the small workspace: a chunk of one block must fit twice in a phase-a source slot
        # (two sub-blocks), as fp32 partials in the partial region and once in an all-gather slot.
        cap = min(_ws_align(max_bytes // 4) // 2, _ws_align(max_bytes // 2) // 2, _ws_align(ag_flag_max_bytes))
        cap = cap // 64 * 64
        self.large_chunk_bytes = max(64, min(cap, 1 << 20 if large_chunk_bytes is None else large_chunk_bytes))
        self.large = os.environ.get("MUSE_TOPO_LARGE", "1") != "0"
        # fused collective + epilogue kernels (all_reduce_boundary, all_gather_boundary, all_gather_rmsnorm)
        self.fused = os.environ.get("MUSE_TOPO_FUSED", "1") != "0"
        self.bf16_exchange = (os.environ.get("MUSE_TOPO_LARGE_BF16_EXCHANGE", "0") == "1"
                              if bf16_exchange is None else bool(bf16_exchange))
        self.ar_large_min = _AR_LARGE_MIN
        self.medium_bx = os.environ.get("MUSE_TOPO_MEDIUM_BX", "1") != "0"
        self.ar_bx_min = _AR_BX_MIN
        self.ar_large_min_bx = _AR_LARGE_MIN_BX
        self.ar_bx_table = _AR_BX_TABLE
        # A bad integer or an unreadable tuning file on one rank must not leave the others blocked in the stage-1
        # all_gather_object: the failure is reported through the stage-1 agreement below.
        tuning_error = ""
        try:
            self.ar_bx_min = int(os.environ.get("MUSE_TOPO_MEDIUM_BX_MIN_BYTES", _AR_BX_MIN))
            self._load_tuning_file(os.environ.get("MUSE_TOPO_AR_TUNING_FILE", ""))
        except Exception as e:
            tuning_error = f"medium all_reduce tuning (MUSE_TOPO_MEDIUM_BX_MIN_BYTES / MUSE_TOPO_AR_TUNING_FILE): {e!r}"
        self.ag_large_min = _AG_LARGE_MIN
        self.rs_min_bytes = _RS_MIN_BYTES
        self._closed = True
        self._ws = None
        if world_size != 8:
            raise TopoUnavailable(f"world size {world_size} != 8")
        if self.device.type != "cuda":
            raise TopoUnavailable("not a CUDA device")
        torch.cuda.set_device(self.device)
        # Every stage ends with an all-rank agreement, so a failure on one rank (missing nvcc, IPC error, ...) makes all
        # ranks raise TopoUnavailable instead of leaving the others blocked in the next collective step.
        # ---- stage 1: topology signature: gather descriptors, derive islands/partners, check P2P -----------------
        try:
            desc = local_descriptor(self.device)
        except Exception as e:  # reported to all ranks through the descriptor
            desc = {"error": repr(e)}
        descs = [None] * world_size
        dist.all_gather_object(descs, desc, group=cpu_group)
        reason = ""
        try:
            self.info: TopoInfo = derive(descs, rank)
            if not self._p2p_ok(descs):
                raise TopoUnavailable("P2P access missing between some GPUs")
        except TopoUnavailable as e:
            reason = str(e)
        except Exception as e:  # e.g. an unparsable sysfs file: report it like any other mismatch, on all ranks
            reason = f"topology derivation failed: {e!r}"
        self._agree(cpu_group, reason or tuning_error, "topology and tuning check")
        # ---- stage 2: JIT build/load, workspace allocation -----------------------------------------------------------
        reason, exc = "", None
        try:
            self.ext = ops.load()
            lay = self.ext.layout(max_bytes, ar1_max_bytes, ag_ll_max_bytes, ag_flag_max_bytes)
            self.workspace_bytes = int(lay["total"])
            self._ws = ops.Workspace(self.ext, self.device, self.workspace_bytes)
        except Exception as e:
            reason, exc = repr(e), e
        self._agree(cpu_group, reason, "extension build / workspace allocation", exc)
        # ---- stage 3: IPC handle exchange and communicator object ------------------------------------------------
        reason, exc = "", None
        try:
            bases = self._ws.exchange(cpu_group, rank, world_size)
            i = self.info
            self._c = self.ext.Comm(rank, i.island, i.ipos, i.opos, i.partner, i.members, i.owners, i.partner_of, bases,
                                    max_bytes, ar1_max_bytes, ag_ll_max_bytes, ag_flag_max_bytes, timeout_s,
                                    self.device.index)
            torch.cuda.synchronize(self.device)
        except Exception as e:
            reason, exc = repr(e), e
        self._agree(cpu_group, reason, "IPC exchange", exc)
        i = self.info
        self._c.set_pdl(os.environ.get("MUSE_TOPO_PDL", "0") == "1")
        # Where the fused collectives load the epilogue operands that do not depend on the collective (BndParams::ld_mode in
        # csrc/topo_types.h). 0 = into registers behind the first push stage: fastest with the operands in HBM (bench_boundary_cold.py,
        # 2026-10-24: all-reduce + boundary at T = 1 / 4 / 8: 12.1 / 14.6 / 18.8 us vs 12.8 / 15.3 / 19.3 for loading after the
        # collective and 12.7 / 16.4 / 20.1 for an L2 prefetch at the start plus loads after the collective).
        self._c.set_ld_mode(int(os.environ.get("MUSE_TOPO_BND_LOADS", "0")))
        self._closed = False
        self._seen: set = set()
        self._stream_ptr = None  # stream of the last eager call
        self._warned_stream = False
        self._strict_stream = os.environ.get("MUSE_TOPO_STRICT_STREAM", "0") == "1"
        logger.info("muse_topo active on rank %d (island %d, partner %d, workspace %.1f MiB): %s", rank, i.island,
                    i.partner, self.workspace_bytes / 2**20, i.summary)

    def _load_tuning_file(self, path: str) -> None:
        """Experiment hook (bench/make_table_medium.py writes the file): JSON with optional keys ``bx_table`` (rows
        of _AR_BX_TABLE), ``bx_min`` (bytes) and ``large_min_bx`` (bytes). Must be the same on every rank (a file that
        cannot be read or parsed on some rank makes every rank raise TopoUnavailable, see ``__init__``)."""
        if not path:
            return
        import json

        with open(path) as f:
            t = json.load(f)
        if "bx_table" in t:
            self.ar_bx_table = [tuple(int(v) for v in row) for row in t["bx_table"]]
        if "bx_min" in t and "MUSE_TOPO_MEDIUM_BX_MIN_BYTES" not in os.environ:
            self.ar_bx_min = int(t["bx_min"])
        if "large_min_bx" in t:
            self.ar_large_min_bx = int(t["large_min_bx"])
        logger.info("muse_topo: all_reduce tuning from %s (bx_min %d, large_min_bx %d, %d table rows)", path,
                    self.ar_bx_min, self.ar_large_min_bx, len(self.ar_bx_table))

    # ------------------------------------------------------------------------------------------------
    def _agree(self, cpu_group, reason: str, what: str, exc: BaseException | None = None) -> None:
        """All ranks report ``reason`` ('' = success); if any rank failed, every rank raises TopoUnavailable."""
        import torch.distributed as dist

        reasons = [None] * self.world_size
        dist.all_gather_object(reasons, reason, group=cpu_group)
        bad = [(r, m) for r, m in enumerate(reasons) if m]
        if not bad:
            return
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        msg = f"{what} failed on rank(s) {[r for r, _ in bad]}: {bad[0][1]}"
        logger.info("muse_topo disabled on rank %d: %s", self.rank, msg)
        raise TopoUnavailable(msg) from exc

    def _p2p_ok(self, descs) -> bool:
        by_bdf = {}
        for idx in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(idx)
            by_bdf[f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"] = idx
        me = self.device.index
        for r, d in enumerate(descs):
            if r == self.rank:
                continue
            idx = by_bdf.get(d["bdf"])
            if idx is None or not torch.cuda.can_device_access_peer(me, idx):
                return False
        return True

    # ------------------------------------------------------------------------------------------------
    def supports(self, op: str, tensor: torch.Tensor, dim: int | None = None) -> bool:
        if self._closed or op not in _OPS:
            return False
        t = tensor
        if not (t.is_cuda and t.device == self.device and t.is_contiguous() and t.numel() > 0):
            return False
        nbytes = t.numel() * t.element_size()
        if op == "all_reduce":
            if t.dtype not in _AR_DTYPES:
                return False
            return (nbytes % 64 == 0 and nbytes <= self.max_bytes) or self._large_ok(nbytes)
        if op == "reduce_scatter":
            return (t.dtype in _AR_DTYPES and t.dim() >= 1 and t.shape[0] % WORLD == 0 and nbytes >= self.rs_min_bytes
                    and self._large_ok(nbytes))
        # all_gather: bit exact data movement, any dtype
        lay = None if dim is None else _ag_layout(t, dim)
        if lay is None:
            return False
        layout, row = lay
        if nbytes % 4 or row % 4:
            return False
        if self._ag_plan(nbytes, layout, row) is not None:
            return True
        return self.large and nbytes % 16 == 0 and (layout == 1 or row % 16 == 0) and nbytes <= _LARGE_MAX_BYTES // 8

    def _large_ok(self, nbytes: int) -> bool:
        """Size conditions of the large all_reduce / reduce_scatter (8 blocks of whole 16-byte packs)."""
        return self.large and nbytes % 128 == 0 and nbytes <= _LARGE_MAX_BYTES

    def _ag_plan(self, shard_bytes: int, layout: int, row_bytes: int = 0):
        """Dispatch row (upto, algorithm, grid, block, unroll) for this shard, or None if the topo path cannot take it."""
        row = _pick(_AG_TABLES[layout], shard_bytes)
        flag_ok = (shard_bytes % 16 == 0 and (layout == 1 or row_bytes % 16 == 0)
                   and shard_bytes <= self.ag_flag_max_bytes)
        ll_ok = shard_bytes <= self.ag_ll_max_bytes
        if (row[1] == 2 and flag_ok) or (row[1] == 1 and ll_ok):
            return row
        if flag_ok:
            return (0, 2, 0, 256, 1)
        if ll_ok:
            return (0, 1, 0, 128, 1)
        return None

    # ------------------------------------------------------------------------------------------------
    def _order_stream(self) -> None:
        """Keep the collectives of this communicator totally ordered (see the stream rule in the docstring)."""
        if torch.cuda.is_current_stream_capturing():
            return
        cur = torch.cuda.current_stream(self.device)
        if self._stream_ptr is None:
            self._stream_ptr, self._stream = cur.cuda_stream, cur
            return
        if cur.cuda_stream == self._stream_ptr:
            return
        if self._strict_stream:
            raise RuntimeError("muse_topo: collectives of one TopoComm must be issued on a single CUDA stream")
        if not self._warned_stream:
            self._warned_stream = True
            logger.warning("muse_topo: collective issued on a second stream; serializing it behind the previous stream "
                           "(logged once)")
        ev = torch.cuda.Event()
        ev.record(self._stream)
        cur.wait_event(ev)
        self._stream_ptr, self._stream = cur.cuda_stream, cur

    def _note(self, op, x, algo):
        """One-time debug summary line per (op, shape, dtype)."""
        key = (op, tuple(x.shape), x.dtype)
        if key not in self._seen and logger.isEnabledFor(logging.DEBUG):
            self._seen.add(key)
            logger.debug("muse_topo rank %d: %s %s %s -> algorithm %s", self.rank, op, tuple(x.shape), x.dtype, algo)

    @staticmethod
    def _stage(x, out):
        """Rank-uniform eligibility: the kernels need 16-byte aligned operands, but alignment is rank-local state, so a
        misaligned input (or a misaligned caller-provided ``out``) is staged through an aligned copy instead of being
        rejected by supports(). Returns (x, out, user_out): ``user_out`` is the caller's ``out`` that must receive a copy."""
        if x.data_ptr() % 16:
            x = x.clone()
        if out is not None and out.data_ptr() % 16:
            return x, None, out
        return x, out, None

    @staticmethod
    def _unstage(res, user_out):
        if user_out is None:
            return res
        user_out.copy_(res)
        return user_out

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        self._order_stream()
        x, out, user_out = self._stage(x, out)
        nbytes = x.numel() * x.element_size()
        large_ar = self.ar_large_min_bx if self._medium_bx(x, nbytes) else self.ar_large_min
        if self._large_ok(nbytes) and (nbytes > self.max_bytes or nbytes >= large_ar):
            self._note("all_reduce", x, "large")
            _, mode, ch, rg, ag, blk, un, crs, cag = _pick(_AR_LARGE_TABLE, nbytes)
            return self._unstage(self._all_reduce_large(x, out, mode, ch, rg, ag, blk, un, None, crs, cag), user_out)
        if self._medium_bx(x, nbytes):
            algo, grid, block, unroll = _pick(self.ar_bx_table, nbytes)[1:]
            self._note("all_reduce", x, "2bx")
            return self._unstage(self._all_reduce(x, algo, block, unroll, MAX_GRID, grid or None, out, bx=True), user_out)
        algo, grid, block, unroll = _pick(_AR_TABLE, nbytes)[1:]
        self._note("all_reduce", x, algo)
        return self._unstage(self._all_reduce(x, algo, block, unroll, MAX_GRID, grid or None, out), user_out)

    def _medium_bx(self, x: torch.Tensor, nbytes: int) -> bool:
        """Whether this all_reduce takes the bf16 exchange of the medium algorithm (bf16 input, the size window of
        ``ar_bx_min`` and up); a function of dtype, size and the construction-time settings only, so it is the same on
        every rank as long as those settings are (see the module docstring)."""
        return self.bf16_exchange and self.medium_bx and x.dtype == torch.bfloat16 and nbytes >= self.ar_bx_min

    def _all_reduce(self, x, algo, block, unroll, max_grid=MAX_GRID, grid=None, out=None, bx=False):
        if out is None:
            out = torch.empty_like(x)
        nbytes = x.numel() * x.element_size()
        items = nbytes // 16 if algo == 1 else nbytes // 64
        if grid is None and algo == 3:
            grid = max(1, min(max_grid, items))
        if grid is None:
            grid = max(1, min(max_grid, _cdiv(items, block * unroll)))
        self._c.all_reduce(x, out, algo, grid, block, unroll, int(bool(bx) and x.dtype == torch.bfloat16))
        return out

    # ------------------------------------------------------------------------------------------------
    # all-reduce fused with the residual boundary of the Muse Spark decoder layer (csrc/boundary.cuh, csrc/ar_bnd.cu)
    # ------------------------------------------------------------------------------------------------
    BOUNDARY_D = 8192
    # What the kernels can serve (one CTA per row; slot capacity of the workspace) ...
    AR_BOUNDARY_MAX_ROWS = 64   # the slice buffers hold 64 rows of a 16 KiB message
    AG_BOUNDARY_MAX_ROWS = 32   # unpooler shards [T, 1024] bf16 (LL8 slots of 64 KiB shards)
    AG_RMS_MAX_ROWS = 64        # pooler shards [T, 512] bf16
    # ... and where they beat the two-step path (bench/bench_boundary.py, 2026-10-24: a single CTA per row serves a few rows well, while
    # the tuned many-CTA kernels of the plain collectives win from 16 rows on: fused - two-step = -1.0 us at T <= 8 and +3.0 us at T = 16
    # for the all-reduce, -1.0 / +0.5 for the unpooler gather, -1.3 us up to T = 16 and +0.5 at T = 32 for the pooler gather).
    DISPATCH_MAX_ROWS = {"all_reduce_boundary": 8, "all_gather_boundary": 8, "all_gather_rmsnorm": 16}

    def _boundary_ok(self, t: torch.Tensor, shape, dtype) -> bool:
        """Rank-uniform operand gate: device, dtype, shape and contiguity only. The kernels need 16-byte aligned operands, but
        alignment is rank-local state (a gate must never depend on it, or one rank could run the two-step path while the others
        run the fused kernel and hang), so misaligned operands are staged through an aligned copy inside the fused calls."""
        return t.is_cuda and t.device == self.device and t.dtype == dtype and tuple(t.shape) == tuple(shape) and t.is_contiguous()

    @staticmethod
    def _aligned(t):
        return t if t.data_ptr() % 16 == 0 else t.clone()

    def supports_boundary_rows(self, kind: str, num_tokens: int) -> bool:
        """Batch-size gate of the fused kernels that the model asks before it has the tensors: ``kind`` is
        "all_reduce_boundary", "all_gather_boundary" or "all_gather_rmsnorm". Where the fused kernel is faster than the two-step path
        (``DISPATCH_MAX_ROWS``). Identical on every rank."""
        if self._closed or not self.fused or kind not in self.DISPATCH_MAX_ROWS:
            return False
        return 1 <= num_tokens <= self.DISPATCH_MAX_ROWS[kind] and self._kernel_rows_ok(kind, num_tokens)

    def _kernel_rows_ok(self, kind: str, num_tokens: int) -> bool:
        """Whether the kernel can serve ``num_tokens`` rows (a hard limit, see AR_BOUNDARY_MAX_ROWS ...)."""
        if kind == "all_reduce_boundary":
            return 1 <= num_tokens <= self.AR_BOUNDARY_MAX_ROWS and num_tokens * 4096 <= _ws_align(self.max_bytes // 4)
        if kind == "all_gather_boundary":
            return 1 <= num_tokens <= self.AG_BOUNDARY_MAX_ROWS
        return kind == "all_gather_rmsnorm" and 1 <= num_tokens <= self.AG_RMS_MAX_ROWS

    def boundary_params_ok(self, alpha, beta, pre_w, post_w=None, out_fp32=None) -> bool:
        """Whether the per-channel operands of a boundary are what the kernels read: fp32 alpha / beta and bf16 weights of
        8192 contiguous elements on this device (``out_fp32``: fp32, contiguous)."""
        D = self.BOUNDARY_D
        ok = (self._boundary_ok(alpha, (D,), torch.float32) and self._boundary_ok(beta, (D,), torch.float32)
              and self._boundary_ok(pre_w, (D,), torch.bfloat16))
        if post_w is not None:
            ok = ok and self._boundary_ok(post_w, (D,), torch.bfloat16)
        if out_fp32 is not None:
            ok = ok and out_fp32.is_cuda and out_fp32.device == self.device and out_fp32.dtype == torch.float32 \
                and out_fp32.is_contiguous()
        return ok

    def supports_all_reduce_boundary(self, x: torch.Tensor, stream: torch.Tensor) -> bool:
        """Whether :meth:`all_reduce_boundary` can serve this call: bf16 ``x`` [T, 8192], 1 <= T <= 64, fp32 ``stream`` of
        the same shape, both contiguous on this communicator's device (rank-uniform: shapes, dtypes, contiguity)."""
        if self._closed or x.dim() != 2 or x.shape[1] != self.BOUNDARY_D:
            return False
        return (not self._closed and self._kernel_rows_ok("all_reduce_boundary", x.shape[0])
                and self._boundary_ok(x, x.shape, torch.bfloat16) and self._boundary_ok(stream, x.shape, torch.float32))

    def supports_all_gather_boundary(self, shard: torch.Tensor, stream: torch.Tensor) -> bool:
        """Whether :meth:`all_gather_boundary` can serve this call: bf16 ``shard`` [T, 1024] (1 <= T <= 32), fp32 ``stream``
        [T, 8192]."""
        if self._closed or shard.dim() != 2 or shard.shape[1] != self.BOUNDARY_D // 8:
            return False
        T = shard.shape[0]
        return (self._kernel_rows_ok("all_gather_boundary", T)
                and self._boundary_ok(shard, shard.shape, torch.bfloat16)
                and self._boundary_ok(stream, (T, self.BOUNDARY_D), torch.float32))

    def supports_all_gather_rmsnorm(self, shard: torch.Tensor) -> bool:
        """Whether :meth:`all_gather_rmsnorm` can serve this call: bf16 ``shard`` [T, 512], 1 <= T <= 64."""
        if self._closed or shard.dim() != 2 or shard.shape[1] != 512:
            return False
        return (self._kernel_rows_ok("all_gather_rmsnorm", shard.shape[0])
                and self._boundary_ok(shard, shard.shape, torch.bfloat16))

    def all_reduce_boundary(self, x, stream, alpha, beta, pre_w, post_w, post_eps, pre_eps, post_gain, pre_gain,
                            out_fp32=None):
        """``all_reduce(x)`` followed by the residual boundary, in one kernel; returns ``(out, new_stream)``.

        With ``y = all_reduce(x)`` (bf16, the arithmetic contract of this module, one final rounding) the kernel computes
        exactly what vLLM's Triton ``_norm_residual_norm_fwd_kernel`` (bT = 1, num_warps = 32, D = 8192, gated residual,
        per-channel alpha / beta) computes from ``y``: ``new_stream = alpha * stream + beta * norm_post(y)`` (fp32) and
        ``out = norm_pre(new_stream)`` rounded to bf16, bit for bit (bench/test_boundary.py, bench/test_ar_boundary.py).
        ``post_w`` None: unweighted post-norm; given: weight-first post-norm (``norm(y * (w + post_gain))``). ``pre_w`` is
        the weight of the pre-norm applied after the normalization with gain ``pre_gain``. ``out_fp32`` (optional fp32
        [T, 8192]) receives ``out`` widened.
        """
        self._order_stream()
        x, stream, alpha, beta, pre_w = (self._aligned(t) for t in (x, stream, alpha, beta, pre_w))
        post_w = None if post_w is None else self._aligned(post_w)
        user_f32, out_fp32 = self._stage_out(out_fp32)
        out = torch.empty_like(x)
        new_stream = torch.empty_like(stream)
        self._c.all_reduce_boundary(x, stream, new_stream, out, out_fp32, alpha, beta, post_w, pre_w, float(post_eps),
                                    float(pre_eps), float(post_gain), float(pre_gain))
        self._unstage_out(user_f32, out_fp32)
        return out, new_stream

    @staticmethod
    def _stage_out(out_fp32):
        """(caller's buffer to copy into or None, buffer the kernel writes): a misaligned ``out_fp32`` goes through a temp."""
        if out_fp32 is None or out_fp32.data_ptr() % 16 == 0:
            return None, out_fp32
        return out_fp32, torch.empty_like(out_fp32)

    @staticmethod
    def _unstage_out(user_f32, staged):
        if user_f32 is not None:
            user_f32.copy_(staged)

    def all_gather_boundary(self, shard, stream, alpha, beta, pre_w, post_w, post_eps, pre_eps, post_gain, pre_gain,
                            out_fp32=None):
        """``all_gather(shard, -1)`` (column concatenation of the 8 unpooler shards [T, 1024] into the branch [T, 8192]) followed
        by the residual boundary, in one kernel; same contract as :meth:`all_reduce_boundary`, returns ``(out, new_stream)``."""
        self._order_stream()
        shard, stream, alpha, beta, pre_w = (self._aligned(t) for t in (shard, stream, alpha, beta, pre_w))
        post_w = None if post_w is None else self._aligned(post_w)
        user_f32, out_fp32 = self._stage_out(out_fp32)
        out = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
        new_stream = torch.empty_like(stream)
        self._c.all_gather_boundary(shard, stream, new_stream, out, out_fp32, alpha, beta, post_w, pre_w, float(post_eps),
                                    float(pre_eps), float(post_gain), float(pre_gain))
        self._unstage_out(user_f32, out_fp32)
        return out, new_stream

    def all_gather_rmsnorm(self, shard, weight, eps, gain):
        """``all_gather(shard, -1)`` of the pooler shards [T, 512] followed by ``pre_expert_norm`` (RMSNorm over 4096, weight
        applied after the normalization, ``f32(weight) + gain``), bf16 [T, 4096], bit for bit the Triton ``_rmsnorm_fwd_kernel``."""
        self._order_stream()
        shard, weight = self._aligned(shard), self._aligned(weight)
        out = torch.empty((shard.shape[0], 4096), dtype=torch.bfloat16, device=shard.device)
        self._c.all_gather_rmsnorm(shard, out, weight, float(eps), float(gain))
        return out

    def all_gather(self, x: torch.Tensor, dim: int, out: torch.Tensor | None = None) -> torch.Tensor:
        self._order_stream()
        lay = _ag_layout(x, dim)
        if lay is None:
            raise ValueError(f"all_gather: dim {dim} of a {x.dim()}-D tensor is not supported by the topo path "
                             "(only dim 0 and the last dim; use supports() as the gate)")
        layout, row = lay
        d = dim if dim >= 0 else dim + x.dim()
        x, out, user_out = self._stage(x, out)
        nbytes = x.numel() * x.element_size()
        large_ok = self.large and nbytes % 16 == 0 and (layout == 1 or row % 16 == 0)
        plan = None if (large_ok and nbytes >= self.ag_large_min) else self._ag_plan(nbytes, layout, row)
        if plan is None:
            if not large_ok:
                raise ValueError("all_gather: shard not supported by the topo path")
            self._note("all_gather" if layout == 0 else "all_gather(dim0)", x, "large")
            return self._unstage(self._all_gather_large(x, layout, out, *_pick(_AGL_TABLE, nbytes)[1:]), user_out)
        self._note("all_gather" if layout == 0 else "all_gather(dim0)", x, plan[1])
        return self._unstage(self._all_gather(x, d, layout, plan[1], plan[3], plan[4], MAX_GRID, plan[2] or None, out),
                             user_out)

    def _all_gather(self, x, d, layout, algo, block, unroll, max_grid=MAX_GRID, grid=None, out=None):
        if out is None:
            shape = list(x.shape)
            shape[-1 if layout == 0 else 0] *= self.world_size
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        nbytes = x.numel() * x.element_size()
        if grid is None:
            if algo == 1:
                grid = max(1, min(max_grid, _cdiv(nbytes // 4, block * unroll)))
            else:  # flag protocol: one chunk of 16 B packs per block
                grid = max(1, min(max_grid, nbytes // 16))
        self._c.all_gather(x, out, layout, algo, grid, block, unroll)
        return out

    def reduce_scatter(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Rows block r of the reduced tensor on rank r: [T, ...] -> [T // 8, ...] (T % 8 == 0)."""
        self._order_stream()
        x, out, user_out = self._stage(x, out)
        nbytes = x.numel() * x.element_size()
        self._note("reduce_scatter", x, "large")
        _, ch, g, blk, un, cut = _pick(_RS_TABLE, nbytes)
        return self._unstage(self._reduce_scatter_large(x, out, ch, g, blk, un, None, cut), user_out)

    # ------------------------------------------------------------------------------------------------
    # large messages (chunked; see the module docstring)
    # ------------------------------------------------------------------------------------------------
    def _chunk(self, chunk_bytes: int) -> int:
        return max(64, min(chunk_bytes, self.large_chunk_bytes))

    def _all_reduce_large(self, x, out=None, mode=1, chunk_bytes=1 << 20, rs_grid=8, ag_grid=24, block=512,
                          unroll=1, bx=None, cut_rs=0, cut_ag=0, ramp=0, pdl=None):
        if out is None:
            out = torch.empty_like(x)
        bx = self.bf16_exchange if bx is None else bx
        if pdl is not None:
            self._c.set_pdl(bool(pdl))
        self._c.all_reduce_large(x, out, self._chunk(chunk_bytes), mode, rs_grid, ag_grid, block, unroll,
                                 int(bool(bx) and x.dtype == torch.bfloat16), int(cut_rs), int(cut_ag), int(ramp))
        if pdl is not None:
            self._c.set_pdl(os.environ.get("MUSE_TOPO_PDL", "0") == "1")
        return out

    def _reduce_scatter_large(self, x, out=None, chunk_bytes=1 << 20, grid=8, block=256, unroll=1, bx=None, cut=0):
        if out is None:
            out = torch.empty((x.shape[0] // WORLD,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        bx = self.bf16_exchange if bx is None else bx
        self._c.reduce_scatter(x, out, self._chunk(chunk_bytes), grid, block, unroll,
                               int(bool(bx) and x.dtype == torch.bfloat16), int(cut))
        return out

    def _all_gather_large(self, x, layout, out=None, chunk_bytes=1 << 20, grid=32, block=256, cut=0):
        if out is None:
            shape = list(x.shape)
            shape[-1 if layout == 0 else 0] *= self.world_size
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        self._c.all_gather_large(x, out, layout, self._chunk(chunk_bytes), grid, block, int(cut))
        return out

    # ------------------------------------------------------------------------------------------------
    def error(self) -> int:
        """Nonzero after a spin timeout (code << 8 | rank + 1); the CUDA context is dead by then."""
        return self._c.error()

    def set_timeout(self, seconds: float) -> None:
        self._c.set_timeout(seconds)

    def set_pdl(self, on: bool) -> None:
        """Launch the kernels with programmatic stream serialization (must be the same on all ranks)."""
        self._c.set_pdl(on)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        self._c.close()
        if self._ws is not None:
            self._ws.close()
            self._ws = None
