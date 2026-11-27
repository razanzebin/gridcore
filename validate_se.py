"""Validate WLS state estimation + bad-data detection on the IEEE 14-, 30- and 118-bus systems.

1. Solve the true operating point with gridcore's power flow.
2. Create noisy SCADA-style readings: |V|, P and Q injections at every bus, and
   P and Q flows on every branch.
3. Corrupt ONE reading badly (a gross error: a failed meter or tampered data).
4. Run WLS: the chi-squared test fails, the largest normalized residual points at the
   corrupted meter, it is removed, and the estimate is rerun and compared to the truth.
"""
import numpy as np
from gridcore.case import load_case
from gridcore.powerflow import run_powerflow
from gridcore.meters import scada_measurements
from gridcore.estimation import estimate_with_bad_data

rng = np.random.default_rng(14)
for n, bad in ((14, ("P", 8)), (30, ("Qf", 11)), (118, ("P", 40))):
    _, ppc = load_case(n)
    pf = run_powerflow(ppc)
    V, ref, nb = pf["V"], int(pf["ref"][0]), len(pf["V"])
    meas, z, sigma, br = scada_measurements(pf, ppc["branch"], rng)
    ib = meas.index(bad)
    z[ib] += 0.5                                  # gross error: 50 MW / MVAr on a 100 MVA base
    res = estimate_with_bad_data(pf["Y"], z, sigma, ref, meas, br, va_ref=np.angle(V[ref]))
    x_true = np.concatenate([np.delete(np.angle(V), ref), np.abs(V)])
    err_vm = np.max(np.abs(res["x"][nb - 1:] - x_true[nb - 1:]))
    err_va = np.rad2deg(np.max(np.abs(res["x"][:nb - 1] - x_true[:nb - 1])))
    flagged = [meas[i] for i in res["removed"]]
    print(f"IEEE {n}-bus: {len(meas)} measurements, corrupted {bad} -> flagged {flagged}; "
          f"J {res['J_initial']:.0f} before, {res['J']:.1f} after (chi2 limit {res['chi2_threshold']:.1f}); "
          f"max |V| error {err_vm:.4f} pu, max angle error {err_va:.3f} deg")
