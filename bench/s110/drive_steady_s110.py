# S1-10 copy of /raid/vllm-runtime/work/s1-9/L11/g/drive_steady11.py (unchanged: the S1-9 L11 STEADY protocol; prompts are warmed into the GPU prefix cache before the windows,
# so the windows see decode only and no request reads the CPU tier).
"""S1-9 L11 gate row: STEADY step times (engine counters, constant batch) on real prompts, several windows per ramp, named cell groups.

usage: python3 -I drive_steady11.py --port 8100 --model M --cells 1:2,16:1,17:1 --tag cold --out-dir DIR [--mode spec|nonspec] [--win 12] [--windows 2] [--n-out auto|N,N,..]
Protocol of the round-5 / L6 / L7 STEADY rows: c requests (real mix320 chats, greedy, ignore_eos, long outputs) are started, the driver waits until all c run and none
waits, 3 s ramp, then the engine counters (spec_decode drafts / generation tokens) are read before and after each of WINDOWS consecutive windows of WIN seconds:
step_ms = 1000 c dt / d(drafts) (spec) or 1000 c dt / d(generated tokens) (non-spec). --cells is c:ramps,...: a ramp is one start of c requests (the batch is
re-formed per ramp, the windows of one ramp share it). One JSON per cell: DIR/steady_<tag>_c<c>.json with {"completed", "failed", "values": [step_ms per window, ramp-major],
"ramps": [[window values of ramp 1], ...], "cpu": [...], "tok_per_draft": [...], "tag", "c", "t_start", "t_end"}. After the last window the still running requests are ABORTED
(their connections are closed; the server cancels them), so a ramp costs ramp + windows, not the tail of the outputs. --n-out auto sizes max_tokens to twice the tokens one request
generates until the last window ends (from a step-time table of the production profile), so that no request can finish inside a window.
CPU = utime + stime of every process of the server's session (from /proc, clock ticks), divided by the window.
"""
import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HARNESS = "/raid/vllm-runtime/work/sd/harness"
sys.path.insert(0, HARNESS)
from acceptance import MODES as _MODES, load_set  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--model", default="unsloth/Muse-Spark-Open-816B-A42B-NVFP4")
ap.add_argument("--cells", required=True, help="c:ramps,c:ramps,...")
ap.add_argument("--tag", required=True)
ap.add_argument("--n-out", default="auto")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--mode", default="spec")
ap.add_argument("--win", type=float, default=12.0)
ap.add_argument("--windows", type=int, default=2)
a = ap.parse_args()
SPEC = a.mode == "spec"
MODES = dict(_MODES)
CLK = os.sysconf("SC_CLK_TCK")
# step ms of the production speculative profile (STEADY of the L11 microbench)
STEP_MS = [(1, 17.3), (8, 36.0), (16, 55.0), (24, 70.0), (32, 82.0), (40, 95.0), (48, 94.0), (64, 104.0), (96, 128.0), (128, 155.0)]
NS_TOK_S = [(1, 90.0), (64, 17.0), (128, 11.0)]   # non-speculative: generated tokens per request per second


def interp(table, c):
    for (c0, s0), (c1, s1) in zip(table, table[1:]):
        if c <= c1:
            return s0 + (s1 - s0) * max(0.0, (c - c0)) / (c1 - c0)
    return table[-1][1]


def auto_n_out(c):
    span = 3.0 + a.windows * a.win + 8.0
    tok_s = 4.7 * 1000.0 / interp(STEP_MS, c) if SPEC else interp(NS_TOK_S, c)
    return int(2.0 * span * tok_s)


GSM, MTB = load_set("gsm80"), load_set("mtbench")
MIX = [x for pair in zip(GSM, MTB) for x in pair]


class Req:
    def __init__(self, item, n_out, port, model):
        self.item, self.n_out, self.port, self.model = item, n_out, port, model
        self.conn = None
        self.status = None
        self.error = None
        self.done = False
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        body = {"model": self.model, "messages": [{"role": "user", "content": self.item["turns"][0]}], "max_tokens": self.n_out, "ignore_eos": True,
                "chat_template_kwargs": {"reasoning_effort": "high", "current_date": "2026-10-24"}, **MODES["greedy"]}
        try:
            self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1800)
            self.conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = self.conn.getresponse()
            self.status = r.status
            r.read()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        self.done = True

    def abort(self):
        try:
            if self.conn is not None and self.conn.sock is not None:
                self.conn.sock.shutdown(2)
                self.conn.close()
        except Exception:  # noqa: BLE001
            pass


def post(prompt_item, n_out):
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=900)
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt_item["turns"][0]}], "max_tokens": n_out, "ignore_eos": True,
            "chat_template_kwargs": {"reasoning_effort": "high", "current_date": "2026-10-24"}, **MODES["greedy"]}
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    assert r.status == 200, (r.status, data[:300])


def metric_sum(text, names):
    tot = 0.0
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        nm = line.split("{")[0].split(" ")[0]
        if nm in names:
            try:
                tot += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return tot


def raw_metrics():
    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
    conn.request("GET", "/metrics")
    t = conn.getresponse().read().decode()
    conn.close()
    return dict(drafts=metric_sum(t, ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")),
                gen=metric_sum(t, ("vllm:generation_tokens_total", "vllm:generation_tokens")),
                run=metric_sum(t, ("vllm:num_requests_running",)), wait=metric_sum(t, ("vllm:num_requests_waiting",)))


def session_pids():
    out = os.environ.get("SD_OUT", "")
    try:
        sid = open(os.path.join(out, "serve.pid")).read().strip()
    except OSError:
        return {}
    res = {}
    for line in subprocess.run(["ps", "-eo", "pid=,sid=,comm="], capture_output=True, text=True).stdout.splitlines():
        f = line.split(None, 2)
        if len(f) == 3 and f[1] == sid:
            res[int(f[0])] = f[2].strip()
    return res


def cpu_ticks():
    res = {}
    for pid, comm in session_pids().items():
        try:
            s = open("/proc/%d/stat" % pid).read()
            rest = s[s.rindex(")") + 2:].split()
            res[pid] = (comm, int(rest[11]) + int(rest[12]))
        except (OSError, ValueError, IndexError):
            pass
    return res


def cpu_window(c0, c1, dt):
    workers, engine, total = [], 0.0, 0.0
    for pid, (comm, t1) in c1.items():
        if pid not in c0:
            continue
        busy = (t1 - c0[pid][1]) / CLK / dt
        total += busy
        if "Worker" in comm:
            workers.append(busy)
        elif "EngineCor" in comm:
            engine += busy
    return {"workers_busy_cores": statistics.mean(workers) if workers else None, "n_workers": len(workers), "engine_busy_cores": engine, "total_busy_cores": total}


def log(msg):
    print("%s %s" % (time.strftime("%F %T"), msg), flush=True)


def steady_ramp(c, n_out):
    """One ramp: returns ([step_ms per window], [cpu per window], [tok_per_draft per window]) or raises."""
    items = [MIX[i % len(MIX)] for i in range(c)]
    ths = [threading.Thread(target=post, args=(it, 32)) for it in items]   # prompts into the prefix cache (the windows see decode only)
    [t.start() for t in ths]
    [t.join() for t in ths]
    reqs = [Req(it, n_out, a.port, a.model) for it in items]
    [r.th.start() for r in reqs]
    t0 = time.time()
    while time.time() - t0 < 90:
        m = raw_metrics()
        if m["run"] >= c and m["wait"] == 0 and m["gen"] > 0:
            break
        time.sleep(0.2)
    time.sleep(3.0)
    vals, cpus, tpds = [], [], []
    for w in range(a.windows):
        k0 = cpu_ticks()
        m0, ta = raw_metrics(), time.time()
        time.sleep(a.win)
        m1, tb = raw_metrics(), time.time()
        k1 = cpu_ticks()
        dt = tb - ta
        if SPEC:
            dd = m1["drafts"] - m0["drafts"]
            step = 1000.0 * c * dt / dd if dd > 0 else float("nan")
            tpd = (m1["gen"] - m0["gen"]) / dd if dd > 0 else float("nan")
        else:
            dg = m1["gen"] - m0["gen"]
            step = 1000.0 * c * dt / dg if dg > 0 else float("nan")
            tpd = None
        cpu = cpu_window(k0, k1, dt)
        finished = sum(1 for r in reqs if r.done)
        log("STEADY mode=%s tag=%s c=%d w=%d step_ms=%.3f dt=%.2fs tok_per_draft=%s running=%d..%d waiting=%d..%d finished_early=%d cpu_workers=%.3f cpu_engine=%.3f" % (
            a.mode, a.tag, c, w + 1, step, dt, ("%.4f" % tpd) if tpd is not None else "-", m0["run"], m1["run"], m0["wait"], m1["wait"], finished,
            cpu["workers_busy_cores"] or -1, cpu["engine_busy_cores"]))
        if finished or m0["run"] < c or m1["run"] < c:
            step = float("nan")   # the batch was not constant: the window is void
        vals.append(step)
        cpus.append(cpu)
        tpds.append(tpd)
    for r in reqs:
        r.abort()
    deadline = time.time() + 60
    while time.time() < deadline:
        m = raw_metrics()
        if m["run"] == 0 and m["wait"] == 0:
            break
        time.sleep(0.5)
    time.sleep(1.0)
    return vals, cpus, tpds


os.makedirs(a.out_dir, exist_ok=True)
cells = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.cells.split(",")]
nouts = [auto_n_out(c) for c, _ in cells] if a.n_out == "auto" else [int(x) for x in a.n_out.split(",")]
fail = 0
for (c, ramps), n_out in zip(cells, nouts):
    t_start = time.time()
    by_ramp, cpus, tpds = [], [], []
    for _ in range(ramps):
        try:
            v, cpu, tpd = steady_ramp(c, n_out)
            by_ramp.append([x for x in v if x == x])
            cpus += cpu
            tpds += tpd
        except Exception as exc:  # noqa: BLE001
            log("STEADY FAILED tag=%s c=%d: %r" % (a.tag, c, exc))
            fail += 1
    vals = [x for r in by_ramp for x in r]
    res = {"completed": len(vals), "failed": ramps * a.windows - len(vals), "values": vals, "ramps": by_ramp, "cpu": cpus, "tok_per_draft": tpds, "mode": a.mode,
           "win_s": a.win, "windows": a.windows, "tag": a.tag, "c": c, "n_out": n_out, "t_start": t_start, "t_end": time.time()}
    json.dump(res, open(os.path.join(a.out_dir, "steady_%s_c%d.json" % (a.tag, c)), "w"))
    if vals:
        log("STEADY_SUMMARY tag=%s c=%d median=%.3f mean=%.3f vals=%s" % (a.tag, c, statistics.median(vals), statistics.mean(vals), ",".join("%.3f" % v for v in vals)))
    if len(vals) < ramps * a.windows:
        fail += 1
log("STEADY11 DONE tag=%s fail=%d" % (a.tag, fail))
sys.exit(1 if fail else 0)
