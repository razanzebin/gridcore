"""Build the bus admittance matrix (Y-bus) from branch data.

Each branch is a pi-model: series impedance z = r + jx, total line charging b
(split b/2 at each end), and an off-nominal transformer tap t = tau * e^{j*shift}
on the from side.

    I_f = Yff V_f + Yft V_t
    I_t = Ytf V_f + Ytt V_t
with  ys = 1/z,
      Ytt = ys + j b/2
      Yff = Ytt / |t|^2
      Yft = -ys / conj(t)
      Ytf = -ys / t
"""
import numpy as np
from scipy.sparse import csr_matrix
from .case import (F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, GS, BS)


def branch_admittances(branch):
    """Per-branch pi-model terms for in-service branches: (from bus, to bus, Yff, Yft)."""
    on = branch[:, BR_STATUS].real > 0
    br = branch[on]
    f = br[:, F_BUS].real.astype(int)
    t = br[:, T_BUS].real.astype(int)
    ys = 1.0 / (br[:, BR_R].real + 1j * br[:, BR_X].real)
    tap = br[:, TAP].real.copy()
    tap[tap == 0] = 1.0
    tap = tap * np.exp(1j * np.deg2rad(br[:, SHIFT].real))
    Ytt = ys + 1j * br[:, BR_B].real / 2
    return f, t, Ytt / (tap * np.conj(tap)), -ys / np.conj(tap)


def make_ybus(baseMVA, bus, branch):
    nb = bus.shape[0]
    on = branch[:, BR_STATUS].real > 0
    br = branch[on]
    f = br[:, F_BUS].real.astype(int)
    t = br[:, T_BUS].real.astype(int)

    ys = 1.0 / (br[:, BR_R].real + 1j * br[:, BR_X].real)
    bc = br[:, BR_B].real
    tap = br[:, TAP].real.copy()
    tap[tap == 0] = 1.0
    tap = tap * np.exp(1j * np.deg2rad(br[:, SHIFT].real))

    Ytt = ys + 1j * bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -ys / np.conj(tap)
    Ytf = -ys / tap

    # bus shunts (MW and MVAr at V = 1 pu), converted to per-unit admittance
    Ysh = (bus[:, GS].real + 1j * bus[:, BS].real) / baseMVA

    rows = np.concatenate([f, f, t, t])
    cols = np.concatenate([f, t, f, t])
    vals = np.concatenate([Yff, Yft, Ytf, Ytt])
    Y = csr_matrix((vals, (rows, cols)), shape=(nb, nb)).toarray()
    Y[np.diag_indices(nb)] += Ysh
    return Y
