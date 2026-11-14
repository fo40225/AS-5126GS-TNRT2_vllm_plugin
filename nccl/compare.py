"""Compare e2e_bitwise.py dumps pairwise. Usage: compare.py REF.json OTHER.json [...]"""
import json
import sys


def load(p):
    return {r["name"]: r for r in json.load(open(p))["requests"]}


def cmp_pair(ref, oth, label):
    print(f"== {label}")
    ident = 0
    all_eq = True
    gmax = 0.0
    for name, r in ref.items():
        o = oth[name]
        a, b = r["token_ids"], o["token_ids"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same = first is None and len(a) == len(b)
        ident += same
        common = len(a) if first is None else first + 1  # include the divergent position (same prefix)
        mx = 0.0
        eq = True
        for i in range(common):
            da, db = r["logprobs"][i], o["logprobs"][i]
            if da.keys() != db.keys():
                eq = False  # top-5 set differs
            for t in da.keys() & db.keys():
                x, y = da[t][0], db[t][0]
                if x != y:
                    eq = False
                mx = max(mx, abs(x - y))
        pl = ""
        if r["prompt_logprobs"]:
            pm, peq = 0.0, True
            for da, db in zip(r["prompt_logprobs"], o["prompt_logprobs"]):
                if da is None:
                    continue
                for t in da.keys() & db.keys():
                    if da[t][0] != db[t][0]:
                        peq = False
                    pm = max(pm, abs(da[t][0] - db[t][0]))
            pl = f" prompt_lp_max={pm:.3e} prompt_lp_bitwise={peq}"
            eq = eq and peq
        all_eq &= eq
        gmax = max(gmax, mx)
        print(f"  {name:8s} plen={r['prompt_len']:5d} ids_identical={same} first_div={first} "
              f"max|dlp|={mx:.3e} lp_bitwise={eq}{pl}")
    print(f"  SUMMARY ids_identical={ident}/{len(ref)} max|dlp|={gmax:.3e} all_logprobs_bitwise={all_eq}")


def rename(d, pre):
    return {k.replace(pre, "short_"): v for k, v in d.items() if k.startswith(pre)}


if sys.argv[1] == "--inproc":  # compare short_* with each shortR{k}_* inside one dump
    d = load(sys.argv[2])
    base = {k: v for k, v in d.items() if k.startswith("short_")}
    k = 1
    while any(n.startswith(f"shortR{k}_") for n in d):
        cmp_pair(base, rename(d, f"shortR{k}_"), f"{sys.argv[2]} short vs shortR{k}")
        k += 1
    sys.exit()

ref = load(sys.argv[1])
for p in sys.argv[2:]:
    cmp_pair(ref, load(p), f"{sys.argv[1]} vs {p}")
