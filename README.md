# gridcore

[![tests](https://github.com/razanzebin/gridcore/actions/workflows/tests.yml/badge.svg)](https://github.com/razanzebin/gridcore/actions/workflows/tests.yml)

Power system analysis written from scratch in Python and NumPy: AC power flow, short-circuit
(fault) analysis, and weighted least squares state estimation with bad-data detection.
Everything is validated against [pandapower](https://www.pandapower.org/) on IEEE test systems.

pandapower is used **only** to load test cases and to provide reference answers. The Y-bus,
the Newton-Raphson solver, the Z-bus fault calculation, and the state estimator are my own code.

## What's built

| Module | What it does |
| --- | --- |
| `gridcore/ybus.py` | Bus admittance matrix from pi-model branches (series impedance, line charging, off-nominal taps, phase shifts) plus bus shunts |
| `gridcore/powerflow.py` | Newton-Raphson AC power flow in polar form, Jacobian built from complex derivatives |
| `gridcore/shortcircuit.py` | Bolted three-phase fault currents from the Z-bus (I = V / Z_kk) and bus voltages during a fault |
| `gridcore/estimation.py` | WLS state estimation (Gauss-Newton), chi-squared bad-data test, largest-normalized-residual identification and removal |
| `gridcore/meters.py` | Builds SCADA-style noisy measurements (bus voltages, injections, branch flows) for testing |
| `tests/` | 19 automated tests, run on every push by GitHub Actions |

## Results

**Power flow** (`python validate.py`): bus voltages vs. pandapower

| Case | NR iterations | Max voltage magnitude error (pu) | Max angle error (deg) |
| --- | --- | --- | --- |
| IEEE 14-bus | 4 | 4.4e-16 | 2.1e-14 |
| IEEE 30-bus | 4 | 8.9e-16 | 4.9e-14 |
| IEEE 118-bus | 5 | 1.2e-15 | 5.0e-13 |
| IEEE 300-bus | 5 | 2.5e-13 | 2.7e-11 |

**Short circuit** (`python validate_sc.py`): a 7-bus, 20 kV meshed network fed by a 500 MVA
utility source, compared with pandapower's IEC 60909 solver (maximum case, c = 1.1).
Fault currents range from 9.2 to 14.4 kA and match at every bus to within 1.2e-14 kA.

**State estimation** (`python validate_se.py`): noisy SCADA readings, with one meter
deliberately corrupted by a gross error (+50 MW or MVAr on a 100 MVA base)

| Case | Measurements | Corrupted meter | Flagged and removed | Chi-squared J before / after (limit) | Max error after removal |
| --- | --- | --- | --- | --- | --- |
| IEEE 14-bus | 82 | P injection, bus 8 | P injection, bus 8 | 1089 / 48 (81) | 0.0024 pu, 0.18 deg |
| IEEE 30-bus | 172 | Q flow, branch 11 | Q flow, branch 11 | 3696 / 104 (150) | 0.0043 pu, 0.26 deg |
| IEEE 118-bus | 726 | P injection, bus 40 | P injection, bus 40 | 1849 / 486 (566) | 0.0020 pu, 0.11 deg |

(Bus and branch numbers are 0-based internal indices.)

## Debugging notes

1. **Loose reference tolerance.** At first the 14- and 30-bus power flows matched to only ~1e-10.
   pandapower's default convergence tolerance was looser than mine; tightening the reference
   (`tolerance_mva=1e-11`) brought both to ~1e-16.
2. **Missing transformer conductance.** The 118- and 300-bus power flows still disagreed (up to 0.012 deg).
   Checking the mismatch bus by bus showed **only real power (P)** was off, and only at transformer
   buses, while reactive power matched. That pointed to a missing shunt *conductance*: pandapower's default
   T-model transformer drops a conductance term when exported to the standard pi-branch table. Using the
   pi transformer model in both the reference and the export fixed it.
3. **30-degree slack-bus angle.** The state estimator matched the 14- and 30-bus cases but was off by
   exactly 30 degrees at every bus on the 118-bus case. IEEE 118's slack bus has a 30-degree reference angle,
   and the estimator assumed 0. Voltages, injections and flows don't change if every angle shifts together,
   so the fit looked perfect while every angle was wrong. Passing the reference angle into the estimator
   brought the error from 30.1 deg to 0.11 deg, and a test now guards it.

## Run it

```bash
pip install -r requirements.txt
python validate.py      # power flow
python validate_sc.py   # short circuit
python validate_se.py   # state estimation + bad data
python -m pytest -q     # 19 tests
```

## Next

- Unsymmetrical faults (line-to-ground, line-to-line) with symmetrical components
- Protective device coordination checks from the fault currents
