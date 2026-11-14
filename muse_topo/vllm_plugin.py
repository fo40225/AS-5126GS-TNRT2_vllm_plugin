"""vLLM plugin: route the tensor-parallel collectives of the 8x RTX PRO 6000 box through ``muse_topo.TopoComm``.

Discovery: ``muse_topo-0.1.dist-info/entry_points.txt`` registers :func:`register` in the ``vllm.general_plugins``
group, so putting the host folder on ``PYTHONPATH`` is enough (no pip install). vLLM calls it in every process
(``load_general_plugins``), in the workers before ``init_device`` creates the device communicators.

``register()`` patches ``CudaCommunicator`` (class level, once per process):

* ``__init__``: after the original constructor, the TP group (``unique_name`` starts with ``tp``) of an 8-GPU world
  creates a ``TopoComm`` (a collective constructor over the group's gloo ``cpu_group``; the IPC workspace is allocated
  here, i.e. during ``init_worker_distributed_environment`` and therefore BEFORE vLLM takes its memory snapshot
  (``init_snapshot``). It is part of the baseline, so it is not charged to the KV budget (``non_kv_cache_memory`` only
  counts what is allocated after the snapshot); it shrinks the ``1 - gpu_memory_utilization`` headroom of the GPU by its
  size, like the other allocations of the distributed initialization). The result is stored in
  ``communicator.topo_comm`` (``None`` when the host does not match, ``MUSE_TOPO_ENABLE != 1`` or the JIT build fails;
  every rank takes the same decision because ``TopoComm`` agrees on failures).
* ``all_reduce`` / ``all_gather`` / ``reduce_scatter``: ``topo_comm.supports(...)`` is the only gate. When it holds the
  topo kernel runs (asynchronous on the current stream, CUDA-graph safe, requested layout produced directly),
  otherwise the original vLLM method (NCCL) runs unchanged. The gate depends on dtype / shape / contiguity only, never on
  rank-local state like the data pointer (misaligned operands are staged through an aligned copy inside ``TopoComm``), so
  all ranks of a collective take the same path. Teardown must be symmetric: ``destroy`` of one rank while the others
  still issue topo collectives would leave them waiting (until ``MUSE_TOPO_TIMEOUT_S``).
* ``destroy``: dumps the call counters and closes the topo communicator.

Microbatched prefill (vLLM ``VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP``): the microbatch threads run their TP collectives on a
comm stream, all of them on the one stream in a rank-uniform order, so the single TopoComm needs no second workspace (its
call counter lives in device memory and the stream rule below is met by the hand-off events). The result of such a
collective is allocated on the compute stream (``_output_stream``): a result allocated on the comm stream would live in a
second allocator pool that grows next to the compute pool and, on a GPU that is nearly full, makes the allocator free and
re-allocate its blocks over and over (about 200 cudaMalloc/cudaFree and 400 event syncs per prefill step).

The sequence-parallel helpers of the Muse model (``sp_all_gather`` / ``sp_reduce_scatter``) first ask the device
communicator's ``custom_all_gather`` / ``custom_reduce_scatter`` (vLLM's CustomAllreduce) and then fall back to
``tensor_model_parallel_all_gather`` / ``_reduce_scatter``, which end in the patched methods above. On this box the
CustomAllreduce is disabled (more than two PCIe-only GPUs), so the first step returns ``None`` and every such call reaches
the topo path through the fallback; the two hooks are therefore not patched.

Environment (all optional):

``MUSE_TOPO_ENABLE=1``       required (checked here and by ``TopoComm``); anything else makes ``register`` a no-op
``MUSE_TOPO_OPS``            comma list of ops to route, default ``all_reduce,all_gather,reduce_scatter``
``MUSE_TOPO_TIMEOUT_S``      spin timeout of the kernels, default 600 s here (TopoComm alone: 10 s). It must exceed
                             the largest start-up skew between ranks (per-rank JIT/Triton compiles run while the
                             other ranks already wait inside a collective); on expiry the kernel traps
``MUSE_TOPO_MAX_BYTES``      largest all-reduce message of the decode/medium algorithms (sets the workspace size), only
                             passed to ``TopoComm`` when set (its default of 8 MiB applies). Larger all-reduces (up to
                             2 GiB: the prefill attn-out / MoE-out reductions), ``reduce_scatter`` and big all-gathers
                             take the chunked large algorithms through the same workspace
``MUSE_TOPO_LARGE``          0 disables the large algorithms (all-reduce above ``max_bytes``, reduce_scatter and
                             all-gather shards above 1 MiB then go to NCCL); default 1
``MUSE_TOPO_LARGE_BF16_EXCHANGE``  1 (opt-in, class deterministic_different): the large all-reduce / reduce_scatter
                             exchange both island partials rounded to bf16 (bf16 inputs only; fp16 / fp32 keep the exact
                             exchange); default 0 (exact fp32 partials). Batch invariant only inside the large algorithm
``MUSE_TOPO_LOG_OPS``        0 (default): the per-op summary goes to the debug log; 1: info log on group rank 0;
                             2: info log on every rank

Per-op summary: the first call of every (op, dim, shape, dtype, stream, capturing, path) logs one line
``muse_topo op[rank r]: all_reduce shape=(1, 8192) bf16 stream=0x.. capturing=False -> topo`` (or ``-> nccl
(reason)`` for calls the topo path cannot take), and ``destroy`` logs the call count of each key. Calls replayed from
a CUDA graph run no Python, so the counts are the number of eager calls plus graph captures.

The plugin never edits vLLM sources; ``VLLM_PLUGINS``, if set, must contain ``muse_topo``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm.muse_topo.plugin")

_PATCH_MARK = "_muse_topo_patched"
_ALL_OPS = ("all_reduce", "all_gather", "reduce_scatter")
_DEFAULT_TIMEOUT_S = 600.0


def _enabled() -> bool:
    return os.environ.get("MUSE_TOPO_ENABLE", "0") == "1"


def _routed_ops() -> frozenset[str]:
    raw = os.environ.get("MUSE_TOPO_OPS")
    if raw is None or not raw.strip():
        return frozenset(_ALL_OPS)
    ops = frozenset(o.strip() for o in raw.split(",") if o.strip())
    unknown = ops - set(_ALL_OPS)
    if unknown:
        logger.warning("muse_topo: ignoring unknown ops in MUSE_TOPO_OPS: %s", sorted(unknown))
    return ops & set(_ALL_OPS)


# ------------------------------------------------------------------------------------------------
# per-op summary
# ------------------------------------------------------------------------------------------------
class _OpTrace:
    """One-time per-(op, dim, shape, dtype, stream, capturing, path) log and call counters of one communicator."""

    def __init__(self, rank: int):
        self.rank = rank
        mode = os.environ.get("MUSE_TOPO_LOG_OPS", "0")
        self._info = mode == "2" or (mode == "1" and rank == 0)
        self.counts: dict[tuple, int] = {}

    def note(self, op: str, x, dim, path: str, reason="") -> None:
        """Count one call; ``reason`` (str or zero-argument callable) is only evaluated for the first call of a key."""
        import torch

        stream = torch.cuda.current_stream(x.device).cuda_stream
        capturing = torch.cuda.is_current_stream_capturing()
        key = (op, dim, tuple(x.shape), x.dtype, stream, capturing, path)
        n = self.counts.get(key)
        if n is not None:
            self.counts[key] = n + 1
            return
        self.counts[key] = 1
        if callable(reason):
            reason = reason()
        log = logger.info if self._info else logger.debug
        log("muse_topo op[rank %d]: %s%s shape=%s %s stream=0x%x capturing=%s -> %s%s", self.rank, op,
            "" if dim is None else f"(dim={dim})", tuple(x.shape), str(x.dtype).replace("torch.", ""), stream,
            capturing, path, f" ({reason})" if reason else "")

    def dump(self) -> None:
        if not self.counts:
            return
        log = logger.info if self._info else logger.debug
        for (op, dim, shape, dtype, stream, capturing, path), n in sorted(self.counts.items(),
                                                                          key=lambda kv: str(kv[0])):
            log("muse_topo calls[rank %d]: %s%s shape=%s %s stream=0x%x capturing=%s -> %s: %d", self.rank, op,
                "" if dim is None else f"(dim={dim})", shape, str(dtype).replace("torch.", ""), stream, capturing, path,
                n)


def _why_not(comm, op: str, t, dim) -> str:
    """Human-readable reason why ``comm.supports`` rejected a call (evaluated lazily, once per key)."""
    import torch

    if getattr(comm, "_closed", False):
        return "communicator closed"
    if not t.is_cuda or t.device != comm.device:
        return f"device {t.device}"
    if t.numel() == 0:
        return "empty tensor"
    if not t.is_contiguous():
        return "non-contiguous"
    nbytes = t.numel() * t.element_size()
    large = getattr(comm, "large", True)
    large_max = 2 << 30  # TopoComm: the large all_reduce / reduce_scatter algorithms stop at 2 GiB
    if op in ("all_reduce", "reduce_scatter"):
        if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"dtype {t.dtype}"
        if op == "reduce_scatter":
            if t.shape[0] % 8:
                return f"{t.shape[0]} rows is not a multiple of 8"
            if nbytes < getattr(comm, "rs_min_bytes", 0):
                return f"{nbytes} B < reduce_scatter minimum {comm.rs_min_bytes} B"
        elif nbytes % 64:
            return f"{nbytes} B is not a multiple of 64"
        elif nbytes <= getattr(comm, "max_bytes", nbytes):
            return "unsupported"
        if not large:
            return f"{nbytes} B needs the large algorithms (MUSE_TOPO_LARGE=0, max_bytes {comm.max_bytes})"
        if nbytes % 128:
            return f"{nbytes} B is not a multiple of 128 (large algorithms)"
        if nbytes > large_max:
            return f"{nbytes} B > {large_max} B (large algorithms)"
    elif op == "all_gather":
        return f"shard of {nbytes} B / dim {dim} is outside the topo all_gather envelope"
    else:
        return f"{op} not supported for {nbytes} B"
    return "unsupported"


# ------------------------------------------------------------------------------------------------
# communicator creation
# ------------------------------------------------------------------------------------------------
def _attach(self, routed: frozenset[str]) -> None:
    """Create the TopoComm of ``self`` (a CudaCommunicator) if it is the 8-GPU TP group."""
    self.topo_comm = None
    self.topo_trace = None
    self.topo_ops = frozenset()
    if self.unique_name.split(":")[0] != "tp" or self.world_size != 8:
        return
    try:
        from muse_topo import TopoComm, TopoUnavailable
    except Exception as e:  # identical on every rank (same environment)
        logger.warning("muse_topo disabled on rank %d: import failed: %r", self.rank, e)
        return
    kwargs = {"timeout_s": float(os.environ.get("MUSE_TOPO_TIMEOUT_S", _DEFAULT_TIMEOUT_S))}
    if os.environ.get("MUSE_TOPO_MAX_BYTES"):
        kwargs["max_bytes"] = int(os.environ["MUSE_TOPO_MAX_BYTES"])
    try:
        comm = TopoComm(self.cpu_group, self.device, self.rank, self.world_size, **kwargs)
    except TopoUnavailable as e:
        logger.info("muse_topo not used for group %s (rank %d): %s", self.unique_name, self.rank, e)
        return
    except Exception:  # TopoComm agrees on its own failures, so this is an unexpected bug: keep vLLM's collectives
        logger.exception("muse_topo disabled on rank %d: creating the TopoComm for group %s failed", self.rank,
                         self.unique_name)
        return
    self.topo_comm = comm
    self.topo_trace = _OpTrace(self.rank)
    self.topo_ops = routed
    logger.info("muse_topo plugin: TopoComm active for group %s on rank %d (ops=%s, timeout %.0f s, workspace %.1f "
                "MiB, max all_reduce of the medium path %d B, large algorithms %s, large bf16 exchange %s)",
                self.unique_name, self.rank, ",".join(sorted(routed)), kwargs["timeout_s"],
                getattr(comm, "workspace_bytes", 0) / 2**20, getattr(comm, "max_bytes", 0),
                "on" if getattr(comm, "large", False) else "off",
                "ON (deterministic_different)" if getattr(comm, "bf16_exchange", False) else "off")


# ------------------------------------------------------------------------------------------------
# wrappers
# ------------------------------------------------------------------------------------------------
def _output_stream():
    """The compute stream on which the result of this collective is to be allocated, or None.

    Set by vLLM's microbatched prefill (VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP) while a collective runs on its comm
    stream: the result is then allocated on the compute stream (the pool the model allocates from) instead of the
    comm stream, which would grow a second pool next to it. The topo kernels write it from the comm stream; the
    compute stream only reads it after the collective and reuses it after those reads, so no extra ordering is needed.
    """
    try:
        from vllm.v1.worker.ubatching import dbo_collective_output_stream
    except ImportError:  # a vLLM without microbatching
        return None
    return dbo_collective_output_stream()


def _alloc(stream, shape, like):
    import torch

    with torch.cuda.stream(stream):
        return torch.empty(shape, dtype=like.dtype, device=like.device)


def _wrap_init(orig, routed):
    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _attach(self, routed)

    return __init__


def _wrap_all_reduce(orig):
    def all_reduce(self, input_):
        comm = getattr(self, "topo_comm", None)
        if comm is not None and "all_reduce" in self.topo_ops:
            if comm.supports("all_reduce", input_):
                self.topo_trace.note("all_reduce", input_, None, "topo")
                stream = _output_stream()
                if stream is not None:
                    return comm.all_reduce(input_, out=_alloc(stream, input_.shape, input_))
                return comm.all_reduce(input_)
            self.topo_trace.note("all_reduce", input_, None, "nccl", lambda: _why_not(comm, "all_reduce", input_, None))
        return orig(self, input_)

    return all_reduce


def _wrap_all_gather(orig):
    def all_gather(self, input_, dim=-1):
        comm = getattr(self, "topo_comm", None)
        if comm is not None and "all_gather" in self.topo_ops:
            d = dim + input_.dim() if dim < 0 else dim
            if comm.supports("all_gather", input_, d):
                self.topo_trace.note("all_gather", input_, d, "topo")
                stream = _output_stream()
                if stream is not None:
                    # column concatenation for the last dim of a >= 2-D tensor, rank-major otherwise (TopoComm layouts)
                    shape = list(input_.shape)
                    shape[-1 if (input_.dim() > 1 and d == input_.dim() - 1) else 0] *= comm.world_size
                    return comm.all_gather(input_, d, out=_alloc(stream, shape, input_))
                return comm.all_gather(input_, d)
            self.topo_trace.note("all_gather", input_, d, "nccl", lambda: _why_not(comm, "all_gather", input_, d))
        return orig(self, input_, dim)

    return all_gather


def _wrap_reduce_scatter(orig):
    def reduce_scatter(self, input_, dim=-1):
        comm = getattr(self, "topo_comm", None)
        if comm is not None and "reduce_scatter" in self.topo_ops:
            d = dim + input_.dim() if dim < 0 else dim
            # TopoComm.reduce_scatter hands out row blocks (dim 0); other dims keep the NCCL path.
            if d == 0 and comm.supports("reduce_scatter", input_, 0):
                self.topo_trace.note("reduce_scatter", input_, 0, "topo")
                stream = _output_stream()
                if stream is not None:
                    shape = [input_.shape[0] // comm.world_size, *input_.shape[1:]]
                    return comm.reduce_scatter(input_, out=_alloc(stream, shape, input_))
                return comm.reduce_scatter(input_)
            self.topo_trace.note("reduce_scatter", input_, d, "nccl",
                                 lambda: "dim != 0" if d != 0 else _why_not(comm, "reduce_scatter", input_, 0))
        return orig(self, input_, dim)

    return reduce_scatter


def _wrap_destroy(orig):
    def destroy(self):
        comm = getattr(self, "topo_comm", None)
        if comm is not None:
            self.topo_comm = None  # later calls fall through to NCCL
            try:
                self.topo_trace.dump()
                comm.close()
            except Exception:
                logger.exception("muse_topo: closing the communicator failed")
        orig(self)

    return destroy


def register() -> None:
    """Entry point of the ``vllm.general_plugins`` group; idempotent, never raises."""
    if not _enabled():
        logger.debug("muse_topo plugin: MUSE_TOPO_ENABLE != 1, not patching vLLM")
        return
    try:
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

        if CudaCommunicator.__dict__.get(_PATCH_MARK):
            return
        routed = _routed_ops()
        CudaCommunicator.__init__ = _wrap_init(CudaCommunicator.__init__, routed)
        CudaCommunicator.all_reduce = _wrap_all_reduce(CudaCommunicator.all_reduce)
        CudaCommunicator.all_gather = _wrap_all_gather(CudaCommunicator.all_gather)
        CudaCommunicator.reduce_scatter = _wrap_reduce_scatter(CudaCommunicator.reduce_scatter)
        CudaCommunicator.destroy = _wrap_destroy(CudaCommunicator.destroy)
        setattr(CudaCommunicator, _PATCH_MARK, True)
        logger.info("muse_topo plugin registered in pid %d (ops=%s)", os.getpid(), ",".join(sorted(routed)))
    except Exception:
        logger.exception("muse_topo plugin: patching CudaCommunicator failed; vLLM keeps its own collectives")
