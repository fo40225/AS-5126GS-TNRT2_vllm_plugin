"""muse_topo: topology-specialized P2P collectives for the 8x RTX PRO 6000 (AS-5126GS-TNRT2) host.

Public API (stable): :class:`TopoComm` (see its docstring), :class:`TopoUnavailable`.
"""

from .collectives import TopoComm
from .topology import TopoUnavailable

__all__ = ["TopoComm", "TopoUnavailable"]
