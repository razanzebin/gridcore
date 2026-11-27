"""Build a realistic SCADA-style measurement set from a solved power flow (for testing the estimator)."""
import numpy as np
from .ybus import branch_admittances


def scada_measurements(r, branch, rng, sigma_v=0.004, sigma_pq=0.01, sigma_flow=0.008):
    """|V|, P and Q injections at every bus plus P and Q flow at the from-end of every branch,
    each with Gaussian meter noise. Returns (meas, z_noisy, sigma, branches)."""
    Y, V = r["Y"], r["V"]
    S = V * np.conj(Y @ V)
    br = branch_admittances(branch)
    f, t, Yff, Yft = br
    Sf = V[f] * np.conj(Yff * V[f] + Yft * V[t])
    meas, z, s = [], [], []
    for b in range(len(V)):
        meas += [("V", b), ("P", b), ("Q", b)]
        z += [abs(V[b]), S[b].real, S[b].imag]
        s += [sigma_v, sigma_pq, sigma_pq]
    for k in range(len(f)):
        meas += [("Pf", k), ("Qf", k)]
        z += [Sf[k].real, Sf[k].imag]
        s += [sigma_flow, sigma_flow]
    s = np.array(s)
    return meas, np.array(z) + rng.normal(0, s), s, br
