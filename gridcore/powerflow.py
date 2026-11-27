"""Newton-Raphson AC power flow in polar coordinates, written from scratch.

Unknowns: voltage angle at every PV and PQ bus, voltage magnitude at every PQ bus.
Mismatch:  dS = V * conj(Y V) - S_spec   (per-unit)
Iterate:   J * [d_theta; d|V|/|V|] = -[dP; dQ]   until max mismatch < tol.
"""
import numpy as np
from .case import (BUS_TYPE, PD, QD, VM, VA, GEN_BUS, PG, QG, VG, GEN_STATUS, PQ, PV, REF)
from .ybus import make_ybus


def _jacobian_blocks(Y, V):
    """dS/dtheta and dS/d|V| (scaled by |V|), the standard complex-form derivatives."""
    I = Y @ V
    diagV, diagI = np.diag(V), np.diag(I)
    diagVnorm = np.diag(V / np.abs(V))
    dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
    dS_dVm = diagV @ np.conj(Y @ diagVnorm) + np.conj(diagI) @ diagVnorm
    return dS_dVa, dS_dVm


def run_powerflow(ppc, tol=1e-10, max_iter=30):
    baseMVA = ppc["baseMVA"]
    bus, gen, branch = ppc["bus"].real, ppc["gen"].real, ppc["branch"]
    nb = bus.shape[0]
    Y = make_ybus(baseMVA, bus, branch)

    ref = np.where(bus[:, BUS_TYPE] == REF)[0]
    pv = np.where(bus[:, BUS_TYPE] == PV)[0]
    pq = np.where(bus[:, BUS_TYPE] == PQ)[0]
    pvpq = np.concatenate([pv, pq])

    # specified complex power injections (gen - load), per-unit
    on = gen[:, GEN_STATUS] > 0
    Sg = np.zeros(nb, dtype=complex)
    np.add.at(Sg, gen[on, GEN_BUS].astype(int), (gen[on, PG] + 1j * gen[on, QG]) / baseMVA)
    Sd = (bus[:, PD] + 1j * bus[:, QD]) / baseMVA
    Sspec = Sg - Sd

    # starting point: flat start, with generator voltage setpoints at PV/REF buses
    Vm = np.ones(nb)
    Va = np.deg2rad(bus[:, VA]) * 0
    Va[ref] = np.deg2rad(bus[ref, VA])          # keep the reference angle
    gbus = gen[on, GEN_BUS].astype(int)
    Vm[gbus] = gen[on, VG]
    V = Vm * np.exp(1j * Va)

    for it in range(1, max_iter + 1):
        mis = V * np.conj(Y @ V) - Sspec
        F = np.concatenate([mis[pvpq].real, mis[pq].imag])
        if np.max(np.abs(F)) < tol:
            return dict(V=V, Y=Y, iterations=it - 1, converged=True, ref=ref, pv=pv, pq=pq)
        dS_dVa, dS_dVm = _jacobian_blocks(Y, V)
        J = np.block([
            [dS_dVa[np.ix_(pvpq, pvpq)].real, dS_dVm[np.ix_(pvpq, pq)].real],
            [dS_dVa[np.ix_(pq, pvpq)].imag,   dS_dVm[np.ix_(pq, pq)].imag],
        ])
        dx = np.linalg.solve(J, -F)
        Va[pvpq] += dx[:len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
        V = Vm * np.exp(1j * Va)
    return dict(V=V, Y=Y, iterations=max_iter, converged=False, ref=ref, pv=pv, pq=pq)
