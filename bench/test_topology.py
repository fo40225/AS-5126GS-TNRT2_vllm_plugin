"""Topology derivation checks (no CUDA context): descriptors come from nvidia-smi, rank orders are permuted.

usage: py_topo.sh bench/test_topology.py
Checks: identity order gives partners 0-4, 1-5, 2-6, 3-7; random permutations of the rank->GPU assignment always
yield a collapse-free mapping (each virtual switch talks to exactly one far switch) with 2 islands x 4 ranks;
cross-check of the switch grouping against `nvidia-smi topo -m` (PIX pairs).
"""
import itertools
import random
import socket
import subprocess

from muse_topo.topology import derive


def smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


rows = [l.split(", ") for l in smi("--query-gpu=index,pci.bus_id,name", "--format=csv,noheader").strip().splitlines()]
host = socket.gethostname()
base = []
for idx, bus, name in rows:
    dom, b, rest = bus.lower().split(":")
    base.append({"host": host, "bdf": f"{dom[-4:]}:{b}:{rest}", "name": name, "cc": (12, 0), "idx": int(idx)})
assert len(base) == 8, base


def run(perm):
    descs = [base[g] for g in perm]  # rank r drives GPU perm[r]
    infos = [derive(descs, r) for r in range(8)]
    return descs, infos


descs, infos = run(list(range(8)))
partners = [i.partner for i in infos]
assert partners == [4, 5, 6, 7, 0, 1, 2, 3], partners
print("identity:", infos[0].summary)

# nvidia-smi topo -m: PIX pairs are the GPUs sharing a virtual switch
topo = smi("topo", "-m").splitlines()
hdr = topo[0].split()
pix = set()
for l in topo[1:9]:
    c = l.split()
    for j, v in enumerate(c[1:9]):
        if v == "PIX":
            pix.add((int(c[0][3:]), j))
sw = {i: infos[i].switches[i] for i in range(8)}
for a, b in itertools.combinations(range(8), 2):
    assert ((a, b) in pix) == (sw[a] == sw[b]), (a, b)
print("topo -m PIX pairs agree with the derived virtual switches:", sorted(p for p in pix if p[0] < p[1]))

rng = random.Random(1)
for t in range(200):
    perm = list(range(8))
    rng.shuffle(perm)
    descs, infos = run(perm)
    far = {}
    for r in range(8):
        i = infos[r]
        assert infos[i.partner].partner == r and i.partner != r
        assert infos[i.partner].island != i.island
        far.setdefault(i.switches[r], set()).add(i.switches[i.partner])
        assert i.owners[i.opos] == r and i.members[i.island][i.ipos] == r
    assert all(len(v) == 1 for v in far.values()), (perm, far)
    # the physical pairing does not depend on the rank assignment: GPU g always pairs with the same GPU
    phys = {perm[r]: perm[infos[r].partner] for r in range(8)}
    assert phys == {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}, phys
print("200 random rank permutations: collapse-free, physical pairing 0-4 1-5 2-6 3-7 stable")
print("TOPOLOGY OK")
