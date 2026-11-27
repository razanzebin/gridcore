"""Load IEEE test cases into plain numpy arrays (MATPOWER-style), in per-unit.

pandapower is used ONLY as a data source for the test systems; every
calculation in gridcore is written from scratch.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.converter.pypower import to_ppc

CASES = {14: pn.case14, 30: pn.case30, 118: pn.case118, 300: pn.case300}

# MATPOWER column indices
BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, BASE_KV = 0, 1, 2, 3, 4, 5, 7, 8, 9
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10
GEN_BUS, PG, QG, VG, GEN_STATUS = 0, 1, 2, 5, 7
PQ, PV, REF = 1, 2, 3


def load_case(n):
    """Return (net, ppc) for IEEE n-bus case. ppc buses are 0-indexed internally."""
    net = CASES[n]()
    pp.runpp(net, tolerance_mva=1e-11, max_iteration=50, trafo_model="pi")  # tight reference, validation only
    ppc = to_ppc(net, init="flat", trafo_model="pi")  # pi-model: exact in branch format
    return net, ppc
