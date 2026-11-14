# S1-7 step-3 diagnosis probe (off unless S17X=1; the process behaves as without this file otherwise).
# A daemon thread in every process that initializes CUDA samples torch's caching-allocator statistics of the device
# it works on at 20 Hz and logs to $S17X_DIR/mem_<pid>.log: current/peak allocated and reserved (MiB), the running
# maxima seen by the sampler, num_alloc_retries (= the allocator hit the wall and released its cache) and num_ooms.
# It only reads CPU-side counters (torch.cuda.memory_stats): no CUDA call, no context creation, no allocation.
import os
import sys
import threading
import time

if os.environ.get("S17X") == "1":
    def _run():
        out_dir = os.environ.get("S17X_DIR", "/tmp")
        while True:
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    break
            except Exception:
                pass
            time.sleep(0.05)
        mib = float(1 << 20)
        n = torch.cuda.device_count()
        f = None
        max_a = max_r = 0.0
        last_logged = (-1.0, -1.0, -1, -1)
        last_t = 0.0
        while True:
            time.sleep(0.05)
            dev = None
            st = None
            for i in range(n):
                try:
                    s = torch.cuda.memory_stats(i)
                except Exception:
                    continue
                if s.get("reserved_bytes.all.peak", 0) > 0 and (st is None or
                        s["reserved_bytes.all.current"] > st["reserved_bytes.all.current"]):
                    dev, st = i, s
            if st is None:
                continue
            a = st["allocated_bytes.all.current"] / mib
            r = st["reserved_bytes.all.current"] / mib
            max_a = max(max_a, a)
            max_r = max(max_r, r)
            ret = st.get("num_alloc_retries", 0)
            oom = st.get("num_ooms", 0)
            now = time.time()
            if (max_a - last_logged[0] >= 16 or max_r - last_logged[1] >= 16 or ret != last_logged[2]
                    or oom != last_logged[3] or now - last_t >= 5.0):
                if f is None:
                    f = open(os.path.join(out_dir, f"mem_{os.getpid()}.log"), "a")
                f.write(f"t={now:.3f} dev={dev} alloc={a:.1f} res={r:.1f} alloc_peak={st['allocated_bytes.all.peak'] / mib:.1f} "
                        f"res_peak={st['reserved_bytes.all.peak'] / mib:.1f} max_alloc_seen={max_a:.1f} "
                        f"max_res_seen={max_r:.1f} retries={ret} ooms={oom}\n")
                f.flush()
                last_logged = (max_a, max_r, ret, oom)
                last_t = now

    threading.Thread(target=_run, name="s17x-mem", daemon=True).start()
# S1-10 diagnosis probe: S17X allocator sampler (see the S1-7 file) plus, when S110_SNAP=1, a CUDA allocator memory-history recording
# (python stacks) in every process that initializes CUDA and a SIGUSR2 handler that dumps the allocator snapshot to
# $S17X_DIR/snap_<pid>.pickle. Off unless the env flags are set; reads/writes nothing else.
import os
import signal
import sys
import threading
import time

if os.environ.get("S110_SNAP") == "1":
    def _enable():
        while True:
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    break
            except Exception:
                pass
            time.sleep(0.02)
        try:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=400000)
        except Exception as e:  # pragma: no cover
            sys.stderr.write("S110_SNAP: record history failed %r\n" % (e,))

    def _dump(signum, frame):
        torch = sys.modules.get("torch")
        out = os.path.join(os.environ.get("S17X_DIR", "/tmp"), "snap_%d.pickle" % os.getpid())
        try:
            torch.cuda.memory._dump_snapshot(out)
        except Exception as e:  # pragma: no cover
            sys.stderr.write("S110_SNAP: dump failed %r\n" % (e,))

    signal.signal(signal.SIGUSR2, _dump)
    threading.Thread(target=_enable, name="s110-snap", daemon=True).start()
