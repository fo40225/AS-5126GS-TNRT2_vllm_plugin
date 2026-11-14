"""Topology signature, islands and partners of the 8x RTX PRO 6000 box (AS-5126GS-TNRT2).

The box has two sockets with four GPUs each. A socket is one *island*; inside an island all pairs are equal, and
between islands traffic must flow only between fixed *partners* so that no virtual PCIe switch talks to two
different far switches at once (that pattern collapses the cross-socket bandwidth ~40x, see docs/topology.md).

Everything is derived from sysfs, so a reordered CUDA_VISIBLE_DEVICES still yields the right islands and partners:
  socket  = physical_package_id of the first CPU of the GPU's NUMA node (the host runs NPS4: 8 NUMA nodes, so the
            NUMA node alone gives 4 groups of 2 GPUs, not the 2 islands)
  switch  = upstream port of the PCIe switch above the GPU (the grandparent in the sysfs device path)
  partner = k-th switch of island 0 (ordered by bus id) <-> k-th switch of island 1, j-th GPU <-> j-th GPU;
            this is 0-4, 1-5, 2-6, 3-7 for CUDA_VISIBLE_DEVICES=0..7 and always collapse-free.

Arithmetic order convention (used by every algorithm): island 0 is the island that holds the smallest rank;
inside an island the ranks are summed in ascending rank order; the final sum is P_0 + P_1.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

import torch


class TopoUnavailable(RuntimeError):
    """The host does not match the signature (or the feature is not enabled); callers fall back to NCCL."""


@dataclass
class TopoInfo:
    rank: int
    world_size: int
    island: int  # 0 or 1 (island 0 holds the smallest rank)
    ipos: int  # position inside the island in arithmetic order (ascending rank)
    opos: int  # position inside the island in partner order (slice ownership of the 3-phase algorithm)
    partner: int
    members: list[list[int]]  # members[I] = ranks of island I, ascending
    owners: list[int]  # my island, partner order: owners[j] owns slice j; partner(owners[j]) has the same j
    partner_of: list[int]
    sockets: list[int]  # socket id per rank
    switches: list[str]  # virtual-switch key per rank
    summary: str = field(default="")


def env_enabled() -> bool:
    return os.environ.get("MUSE_TOPO_ENABLE", "0") == "1"


# ------------------------------------------------------------------------------------------------
# sysfs
# ------------------------------------------------------------------------------------------------
def _read(path: str) -> str:
    return Path(path).read_text().strip()


def _bdf(dev_props) -> str:
    return f"{dev_props.pci_domain_id:04x}:{dev_props.pci_bus_id:02x}:{dev_props.pci_device_id:02x}.0"


def _numa_node(bdf: str) -> int:
    return int(_read(f"/sys/bus/pci/devices/{bdf}/numa_node"))


def _socket_of_numa(node: int) -> int:
    cpulist = _read(f"/sys/devices/system/node/node{node}/cpulist")
    first_cpu = int(cpulist.split(",")[0].split("-")[0])
    return int(_read(f"/sys/devices/system/cpu/cpu{first_cpu}/topology/physical_package_id"))


def _switch_key(bdf: str) -> str:
    """Upstream port of the PCIe switch above the GPU: the grandparent in the sysfs device path."""
    real = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    parts = [p for p in real.split("/") if p.count(":") == 2 and "." in p]  # BDF components only
    if len(parts) < 3:
        raise TopoUnavailable(f"cannot resolve the upstream bridge of {bdf}: {real}")
    return parts[-3]


def local_descriptor(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    bdf = _bdf(props)
    return {"host": socket.gethostname(), "bdf": bdf, "name": props.name, "cc": (props.major, props.minor)}


# ------------------------------------------------------------------------------------------------
# derivation (pure function of the gathered descriptors; sysfs reads are for the bdfs listed)
# ------------------------------------------------------------------------------------------------
def derive(descs: list[dict], rank: int) -> TopoInfo:
    world = len(descs)
    if world != 8:
        raise TopoUnavailable(f"world size {world} != 8")
    for r, d in enumerate(descs):
        if "error" in d:
            raise TopoUnavailable(f"rank {r} could not describe its GPU: {d['error']}")
    if len({d["host"] for d in descs}) != 1:
        raise TopoUnavailable("ranks span several hosts")
    for d in descs:
        if "RTX PRO 6000" not in d["name"] or "Blackwell" not in d["name"] or tuple(d["cc"]) != (12, 0):
            raise TopoUnavailable(f"unsupported GPU {d['name']} cc {d['cc']}")
    try:
        numa = [_numa_node(d["bdf"]) for d in descs]
        if any(n < 0 for n in numa):
            raise TopoUnavailable("a GPU reports numa_node -1")
        sockets = [_socket_of_numa(n) for n in numa]
        switches = [_switch_key(d["bdf"]) for d in descs]
    except OSError as e:
        raise TopoUnavailable(f"sysfs topology not readable: {e}") from e
    # islands = physical sockets (NUMA nodes are finer: NPS4 gives 4 groups of 2 GPUs); island 0 holds rank 0
    sock_ids = sorted(set(sockets), key=lambda s: min(r for r in range(world) if sockets[r] == s))
    if len(sock_ids) != 2 or any(sockets.count(s) != 4 for s in sock_ids):
        raise TopoUnavailable(f"expected 2 sockets x 4 GPUs, got sockets={sockets}")
    members = [sorted(r for r in range(world) if sockets[r] == s) for s in sock_ids]
    # partner order: the k-th virtual switch of island 0 (by bus id) pairs with the k-th switch of island 1, the
    # j-th GPU (by bus id) inside paired switches pairs with the j-th GPU. Ordering by PCI address (not by rank)
    # keeps the physical pairing stable when CUDA_VISIBLE_DEVICES reorders the GPUs.
    plist = []
    for isl in members:
        sw_ids = sorted({switches[r] for r in isl})
        if len(sw_ids) != 2 or any(sum(switches[r] == k for r in isl) != 2 for k in sw_ids):
            raise TopoUnavailable(f"expected 2 virtual switches x 2 GPUs per socket, got {[switches[r] for r in isl]}")
        plist.append([r for k in sw_ids for r in sorted((r for r in isl if switches[r] == k), key=lambda r: descs[r]["bdf"])])
    partner_of = [-1] * world
    for j in range(4):
        partner_of[plist[0][j]] = plist[1][j]
        partner_of[plist[1][j]] = plist[0][j]
    # collapse-free check: the two GPUs of a virtual switch must map to GPUs of one far switch, and the two
    # switches of an island to different far switches
    far = {}
    for r in range(world):
        far.setdefault(switches[r], set()).add(switches[partner_of[r]])
    if any(len(v) != 1 for v in far.values()):
        raise TopoUnavailable("partner mapping would send one virtual switch to two far switches")
    isl = 0 if rank in members[0] else 1
    info = TopoInfo(
        rank=rank, world_size=world, island=isl, ipos=members[isl].index(rank), opos=plist[isl].index(rank),
        partner=partner_of[rank], members=members, owners=plist[isl], partner_of=partner_of, sockets=sockets,
        switches=switches)
    info.summary = (f"islands={members} partner_of={partner_of} owners={plist[isl]} "
                    f"switches={[switches[r][-7:] for r in range(world)]} numa={numa}")
    return info
