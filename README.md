# gridcore

Power system analysis written from scratch in Python and NumPy, validated against
[pandapower](https://www.pandapower.org/) on the IEEE 14-, 30-, 118- and 300-bus test systems.

pandapower is used **only** to load the IEEE test cases and to provide a reference
solution. The Y-bus and the Newton-Raphson solver are my own code.

## What's built

| Module | What it does |
| --- | --- |
| `gridcore/ybus.py` | Builds the bus admittance matrix from pi-model branches (series impedance, line charging, off-nominal taps and phase shifts) plus bus shunts |
| `gridcore/powerflow.py` | Newton-Raphson AC power flow in polar form, with the Jacobian built from complex derivatives |
| `validate.py` | Runs all four IEEE cases and compares bus voltages to pandapower |

## Results

| Case | NR iterations | Max voltage magnitude error (pu) | Max angle error (deg) |
| --- | --- | --- | --- |
| IEEE 14-bus | 4 | 4.4e-16 | 2.1e-14 |
| IEEE 30-bus | 4 | 8.9e-16 | 4.9e-14 |
| IEEE 118-bus | 5 | 1.2e-15 | 5.0e-13 |
| IEEE 300-bus | 5 | 2.5e-13 | 2.7e-11 |

All four match the reference to machine precision.

## Debugging notes

1. **Loose reference tolerance.** At first the 14- and 30-bus cases matched to only ~1e-10.
   pandapower's default convergence tolerance was looser than mine; tightening the reference
   (`tolerance_mva=1e-11`) brought both to ~1e-16.
2. **Missing transformer conductance.** The 118- and 300-bus cases still disagreed (up to 0.012 deg).
   Checking the power mismatch bus by bus showed **only real power (P)** was off, and only at
   transformer buses, while reactive power matched exactly. That pointed to a missing shunt
   *conductance*. pandapower models transformers as a T-circuit by default; exporting that to the
   standard pi-branch table drops a conductance term. Using the pi transformer model consistently in
   both the reference and the export fixed it, and both cases now match to machine precision.

## Run it

```bash
pip install numpy scipy pandapower
python validate.py
```

## Next

- Short-circuit (fault) analysis: three-phase bolted faults from the Z-bus
- Weighted least squares state estimation with bad-data detection
- Unit tests and GitHub Actions CI
