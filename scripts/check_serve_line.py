#!/usr/bin/env python3
"""Post-boot gate of the launch scripts (S1-11 review r2 m1): the serve line the server is REALLY running with is the one the env files derive.

Three legs, all must hold (exit 0), any difference is a FAIL (exit 1):
  1. the live command line of the API-server process (NUL-exact, from /proc/<pid>/cmdline or --cmdline-file; the part from `serve` on, behind the
     interpreter and the vllm script) equals serve_cmd.txt, argument by argument;
  2. with --derived: serve_cmd.txt equals, byte for byte, the serve line a clean `env -i` dry run of the same script derives from the env files at check time;
  3. the "non-default args" line(s) of the startup log, vLLM's own parse of the arguments, carry exactly the options of that command line with the same values
     (scalars, the JSON of --speculative-config / --limit-mm-per-prompt and the three fields of --kv-transfer-config, and no kv_connector / kv_role / non-empty
     kv_connector_extra_config in the log's KVTransferConfig that the command line lacks), and no option that is not on it.
usage: check_serve_line.py --serve-cmd FILE (--pid PID | --cmdline-file FILE) --log FILE [--derived FILE]
"""
import argparse
import ast
import json
import os
import re
import shlex
import sys


def live_argv(args):
    raw = open(args.cmdline_file if args.cmdline_file else "/proc/%s/cmdline" % args.pid, "rb").read()
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return [p.decode("utf-8", "surrogateescape") for p in parts]


def top_entries(body):
    """Split the text of a Python-repr dict at its top-level commas -> {key: value text}."""
    body = body.strip()
    assert body.startswith("{") and body.endswith("}"), "not a dict"
    items, cur, depth, quote, esc = [], [], 0, None, False
    for ch in body[1:-1]:
        if quote:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    items.append("".join(cur))
    out = {}
    for it in items:
        k, sep, v = it.partition(":")
        if not sep:
            raise ValueError("no key in %r" % it[:60])
        out[ast.literal_eval(k.strip())] = v.strip()
    return out


def balanced(text, start):
    """text[start] == '{': the balanced {...} substring."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces")


def parse_serve(argv):
    """['vllm', 'serve', model, --flag value ...] -> (model, {key: (kind, value)}) with kind 'bool' | 'str'."""
    model = argv[2]
    opts, i = {}, 3
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError("unexpected positional argument %r" % tok)
        key = tok[2:].replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opts[key] = ("str", argv[i + 1])
            i += 2
        else:
            opts[key] = ("bool", True)
            i += 1
    return model, opts


def same_scalar(want, got):
    if isinstance(got, bool):
        return want is True and got is True
    if isinstance(got, float):
        try:
            return float(want) == got
        except ValueError:
            return False
    return str(got) == want


def kv_transfer_fields(text, want):
    """KVTransferConfig(...) repr vs the JSON of --kv-transfer-config: returns a list of differences."""
    bad = []
    for k, v in want.items():
        if isinstance(v, dict):
            m = re.search(r"\b%s=(\{)" % re.escape(k), text)
            try:
                got = ast.literal_eval(balanced(text, m.start(1))) if m else None
            except (ValueError, SyntaxError):
                got = None
            if got != v:
                bad.append("%s: line %s, command line %s" % (k, got, v))
        else:
            m = re.search(r"\b%s=('[^']*'|\"[^\"]*\"|[^,)\s]+)" % re.escape(k), text)
            got = ast.literal_eval(m.group(1)) if m else None
            if got != v:
                bad.append("%s: line %r, command line %r" % (k, got, v))
    # a key that is NOT on the command line must not be set in vLLM's parse either (S1-11 review p2 r2 F3): a connector or a role that the line lacks but the log
    # carries means the line was edited while the boot ran another configuration. Only kv_connector and kv_role are read (the other fields of the repr are
    # generated defaults: engine_id, kv_port, ...).
    for k in ("kv_connector", "kv_role"):
        if k in want:
            continue
        m = re.search(r"\b%s=('[^']*'|\"[^\"]*\"|[^,)\s]+)" % re.escape(k), text)
        got = ast.literal_eval(m.group(1)) if m else None
        if got is not None:
            bad.append("%s: line %r, but the command line has no %s" % (k, got, k))
    # the tier size lives in kv_connector_extra_config (dict-valued, default {}): a non-empty dict in the log that the line lacks is the same kind of edit
    if "kv_connector_extra_config" not in want:
        m = re.search(r"\bkv_connector_extra_config=(\{)", text)
        try:
            got = ast.literal_eval(balanced(text, m.start(1))) if m else None
        except (ValueError, SyntaxError):
            got = "unparseable"
        if got:
            bad.append("kv_connector_extra_config: line %s, but the command line has no kv_connector_extra_config" % (got,))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-cmd", required=True)
    ap.add_argument("--pid")
    ap.add_argument("--cmdline-file")
    ap.add_argument("--log", required=True)
    ap.add_argument("--derived")
    args = ap.parse_args()
    if not (args.pid or args.cmdline_file):
        ap.error("--pid or --cmdline-file")
    fails = []
    want = shlex.split(open(args.serve_cmd, encoding="utf-8", errors="surrogateescape").read())
    if want[:2] != ["vllm", "serve"]:
        print("FAIL serve_cmd.txt does not start with `vllm serve`: %r" % want[:3])
        return 1

    # leg 1: the live command line
    try:
        argv = live_argv(args)
    except OSError as e:
        print("FAIL cannot read the live command line: %s" % e)
        return 1
    idx = next((i for i in range(1, len(argv)) if argv[i] == "serve" and os.path.basename(argv[i - 1]) == "vllm"), None)
    if idx is None:
        fails.append("the live command line has no `vllm serve` part: %r" % argv[:4])
        live = []
    else:
        live = ["vllm"] + argv[idx:]
        if live != want:
            d = [i for i in range(max(len(live), len(want))) if (live[i] if i < len(live) else None) != (want[i] if i < len(want) else None)]
            fails.append("live command line differs from serve_cmd.txt at %d argument(s): %s" % (
                len(d), "; ".join("[%d] live %r vs serve_cmd %r" % (i, live[i] if i < len(live) else None, want[i] if i < len(want) else None) for i in d[:4])))

    # leg 2: the clean-shell derivation at check time
    if args.derived:
        if open(args.derived, "rb").read() != open(args.serve_cmd, "rb").read():
            fails.append("serve_cmd.txt differs from the serve line re-derived from the env files in a clean shell (%s)" % args.derived)

    # leg 3: vLLM's parse of the arguments (the startup log)
    model, opts = parse_serve(want)
    lines = [l.split("non-default args:", 1)[1].strip() for l in open(args.log, encoding="utf-8", errors="replace") if "non-default args:" in l]
    if not lines:
        fails.append('no "non-default args" line in the log: the parsed launch arguments cannot be confirmed')
    for n, line in enumerate(lines, 1):
        try:
            ent = top_entries(line)
        except (ValueError, AssertionError, SyntaxError) as e:
            fails.append("non-default args line %d is not parseable (%s)" % (n, e))
            continue
        tag = "non-default args line %d" % n
        for k, v in (("model_tag", model), ("model", model)):
            if k not in ent or ast.literal_eval(ent[k]) != v:
                fails.append("%s: %s is %s, the command line has %r" % (tag, k, ent.get(k), v))
        for key, (kind, val) in opts.items():
            if key not in ent:
                fails.append("%s: --%s is on the command line but not in the parsed arguments" % (tag, key.replace("_", "-")))
                continue
            text = ent[key]
            if kind == "bool":
                ok = text == "True"
            elif val.startswith("{") and key == "kv_transfer_config":
                bad = kv_transfer_fields(text, json.loads(val))
                if bad:
                    fails.append("%s: %s differs: %s" % (tag, key, "; ".join(bad)))
                continue
            elif val.startswith("{"):
                want_d = json.loads(val)
                try:
                    got_d = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    got_d = None
                ok = got_d == want_d
                if not ok and isinstance(got_d, dict):
                    diff = ["%s: line %r, command line %r" % (k, got_d.get(k), want_d.get(k)) for k in sorted(set(got_d) | set(want_d)) if got_d.get(k) != want_d.get(k)]
                    fails.append("%s: %s differs: %s" % (tag, key, "; ".join(diff)))
                    continue
            else:
                try:
                    ok = same_scalar(val, ast.literal_eval(text))
                except (ValueError, SyntaxError):
                    ok = text == val
            if not ok:
                fails.append("%s: %s is %s, the command line has %s" % (tag, key, text[:80], val[:80]))
        extra = sorted(set(ent) - set(opts) - {"model_tag", "model"})
        if extra:
            fails.append("%s carries options that are not on the command line: %s" % (tag, ", ".join(extra)))

    for f in fails:
        print("FAIL " + f)
    if fails:
        return 1
    print("OK live command line = serve_cmd.txt%s; %d non-default args line(s) carry all %d options (and no other)" % (
        " = clean-shell derivation" if args.derived else "", len(lines), len(opts) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
