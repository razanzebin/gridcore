"""Compare gridcore's Newton-Raphson power flow with pandapower on IEEE test systems."""
import numpy as np
from gridcore.case import load_case
from gridcore.powerflow import run_powerflow

print(f"{'case':>8} {'iters':>5} {'max |dVm| (pu)':>16} {'max |dVa| (deg)':>16}")
for n in (14, 30, 118, 300):
    net, ppc = load_case(n)
    r = run_powerflow(ppc)
    # map ppc internal bus order back to pandapower bus indices
    lookup = net._pd2ppc_lookups["bus"]
    idx = lookup[net.res_bus.index.values]
    Vm, Va = np.abs(r["V"][idx]), np.rad2deg(np.angle(r["V"][idx]))
    dVm = np.max(np.abs(Vm - net.res_bus.vm_pu.values))
    dVa = np.max(np.abs(Va - net.res_bus.va_degree.values))
    print(f"{n:>5}-bus {r['iterations']:>5} {dVm:>16.2e} {dVa:>16.2e}  converged={r['converged']}")
