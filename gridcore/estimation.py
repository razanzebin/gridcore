"""Weighted least squares (WLS) state estimation with bad-data detection, written from scratch.

A control center never sees the true grid state; it sees noisy meter readings z.
State estimation finds the bus voltages x (angles + magnitudes) that best explain them:
    minimize  J(x) = sum_i  (z_i - h_i(x))^2 / sigma_i^2
Solved with Gauss-Newton:  (H^T W H) dx = H^T W (z - h(x)),  W = diag(1/sigma^2).

Bad-data detection:
  1. Chi-squared test: if J(x) is larger than expected for (m - n) degrees of freedom,
     at least one measurement is probably bad.
  2. Largest normalized residual: r_N,i = |r_i| / sqrt(Omega_ii), Omega = R - H G^-1 H^T.
     The measurement with the largest r_N (> 3) is flagged, removed, and the estimate rerun.
"""
import numpy as np
from scipy.stats import chi2


def measurement_model(Y, x, ref, meas, branches=None, va_ref=0.0):
    """h(x): predicted measurements for state x = [angles (non-ref buses), |V| (all buses)].
    meas is a list of (type, index):
      "V", "P", "Q"  -> voltage magnitude / real / reactive injection at a bus
      "Pf", "Qf"     -> real / reactive flow at the from-end of a branch (needs branches)
    branches = (f, t, Yff, Yft) from gridcore.ybus.branch_admittances. All per-unit."""
    nb = Y.shape[0]
    va = np.full(nb, va_ref)          # reference bus keeps its known angle (radians)
    nonref = np.setdiff1d(np.arange(nb), ref)
    va[nonref] = x[:len(nonref)]
    vm = x[len(nonref):]
    V = vm * np.exp(1j * va)
    S = V * np.conj(Y @ V)
    if branches is not None:
        f, t, Yff, Yft = branches
        Sf = V[f] * np.conj(Yff * V[f] + Yft * V[t])
    out = np.empty(len(meas))
    for i, (kind, b) in enumerate(meas):
        if kind == "V":    out[i] = vm[b]
        elif kind == "P":  out[i] = S[b].real
        elif kind == "Q":  out[i] = S[b].imag
        elif kind == "Pf": out[i] = Sf[b].real
        elif kind == "Qf": out[i] = Sf[b].imag
    return out


def jacobian(Y, x, ref, meas, branches=None, va_ref=0.0, eps=1e-7):
    """Numerical Jacobian dh/dx (central differences)."""
    H = np.empty((len(meas), len(x)))
    for j in range(len(x)):
        d = np.zeros_like(x); d[j] = eps
        H[:, j] = (measurement_model(Y, x + d, ref, meas, branches, va_ref)
                   - measurement_model(Y, x - d, ref, meas, branches, va_ref)) / (2 * eps)
    return H


def wls(Y, z, sigma, ref, meas, branches=None, va_ref=0.0, tol=1e-8, max_iter=20):
    nb = Y.shape[0]
    x = np.concatenate([np.full(nb - 1, va_ref), np.ones(nb)])  # flat start at the reference angle
    W = np.diag(1.0 / sigma**2)
    for it in range(1, max_iter + 1):
        r = z - measurement_model(Y, x, ref, meas, branches, va_ref)
        H = jacobian(Y, x, ref, meas, branches, va_ref)
        G = H.T @ W @ H                                          # gain matrix
        dx = np.linalg.solve(G, H.T @ W @ r)
        x += dx
        if np.max(np.abs(dx)) < tol:
            break
    r = z - measurement_model(Y, x, ref, meas, branches, va_ref)
    H = jacobian(Y, x, ref, meas, branches, va_ref)
    G = H.T @ W @ H
    J = float(r @ W @ r)
    Omega = np.diag(sigma**2) - H @ np.linalg.solve(G, H.T)     # residual covariance
    rn = np.abs(r) / np.sqrt(np.clip(np.diag(Omega), 1e-30, None))
    return dict(x=x, J=J, rn=rn, iterations=it, dof=len(z) - len(x))


def estimate_with_bad_data(Y, z, sigma, ref, meas, branches=None, va_ref=0.0, alpha=0.01, rn_limit=3.0, max_removals=5):
    """Run WLS, then repeatedly remove the measurement with the largest normalized
    residual while the chi-squared test fails. Returns the final estimate and removed indices."""
    keep = np.arange(len(z))
    removed = []
    J_initial = None
    for _ in range(max_removals + 1):
        m = [meas[i] for i in keep]
        res = wls(Y, z[keep], sigma[keep], ref, m, branches, va_ref)
        threshold = chi2.ppf(1 - alpha, res["dof"])
        res["chi2_threshold"] = threshold
        if J_initial is None:
            J_initial = res["J"]
        if res["J"] <= threshold or res["rn"].max() <= rn_limit:
            break
        worst = int(np.argmax(res["rn"]))
        removed.append(int(keep[worst]))
        keep = np.delete(keep, worst)
    res["removed"] = removed
    res["J_initial"] = J_initial
    res["keep"] = keep
    return res
