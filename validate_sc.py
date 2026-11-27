"""Validate gridcore's Z-bus short-circuit calculation against pandapower's IEC 60909 solver.

Test network: a 7-bus, 20 kV meshed distribution system fed by a utility source
(external grid, Sk'' = 500 MVA, R/X = 0.1). For maximum fault current IEC 60909 uses
a voltage factor c = 1.1 at medium voltage and neglects loads and line charging, so
gridcore uses the same assumptions: prefault voltage c, source impedance c*Un^2/Sk''.
"""
import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
from gridcore.shortcircuit import zbus, bolted_three_phase, pu_to_ka, post_fault_voltages

UN_KV, S_BASE, SK_MVA, RX, C = 20.0, 100.0, 500.0, 0.1, 1.1
# (from, to, length_km, r_ohm_per_km, x_ohm_per_km)
LINES = [(0, 1, 2.0, 0.161, 0.117), (1, 2, 1.5, 0.161, 0.117), (2, 3, 3.0, 0.206, 0.123),
         (3, 4, 1.0, 0.206, 0.123), (1, 5, 2.5, 0.161, 0.117), (5, 6, 1.2, 0.206, 0.123),
         (6, 3, 2.2, 0.206, 0.123), (4, 0, 4.0, 0.161, 0.117)]
NB = 7

def reference():
    net = pp.create_empty_network(sn_mva=S_BASE)
    for i in range(NB):
        pp.create_bus(net, vn_kv=UN_KV)
    pp.create_ext_grid(net, 0, s_sc_max_mva=SK_MVA, rx_max=RX)
    for f, t, l, r, x in LINES:
        pp.create_line_from_parameters(net, f, t, l, r_ohm_per_km=r, x_ohm_per_km=x,
                                       c_nf_per_km=0.0, max_i_ka=1.0)
    sc.calc_sc(net, case="max", ip=False, ith=False)
    return net.res_bus_sc.ikss_ka.values

def gridcore_sc():
    z_base = UN_KV**2 / S_BASE                       # ohms
    Y = np.zeros((NB, NB), dtype=complex)
    for f, t, l, r, x in LINES:
        y = 1.0 / (complex(r, x) * l / z_base)
        Y[f, f] += y; Y[t, t] += y; Y[f, t] -= y; Y[t, f] -= y
    zq = C * UN_KV**2 / SK_MVA                        # source impedance magnitude, ohms
    xq = zq / np.sqrt(1 + RX**2)
    ysrc = np.zeros(NB, dtype=complex)
    ysrc[0] = 1.0 / (complex(RX * xq, xq) / z_base)
    Z = zbus(Y, ysrc)
    return pu_to_ka(bolted_three_phase(Z, C), S_BASE, UN_KV), Z

if __name__ == "__main__":
    ref = reference()
    mine, Z = gridcore_sc()
    print(f"{'bus':>4} {'gridcore (kA)':>14} {'pandapower (kA)':>16} {'diff':>10}")
    for k in range(NB):
        print(f"{k:>4} {mine[k]:>14.6f} {ref[k]:>16.6f} {abs(mine[k]-ref[k]):>10.2e}")
    print(f"max |difference| = {np.max(np.abs(mine-ref)):.2e} kA")
    v = np.abs(post_fault_voltages(Z, 3, C))
    print("voltages during a fault at bus 3 (pu):", np.round(v, 3))
