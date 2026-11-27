"""Three-phase bolted fault analysis from the bus impedance matrix (Z-bus), written from scratch.

For a bolted three-phase fault at bus k (fault impedance = 0), Thevenin's theorem gives
    I_fault(k) = V_prefault(k) / Z_kk
where Z = Y^-1 is the bus impedance matrix of the network *including the sources*
(utility/external-grid equivalents or generator subtransient reactances).
The diagonal Z_kk is the Thevenin impedance seen looking into the network at bus k.

During a fault at k, the voltage at every other bus i drops to
    V_i = V_pre(i) - Z_ik * I_fault(k)
which is how you see how far a fault "pulls down" the rest of the system.
"""
import numpy as np


def zbus(Y_network, source_admittance):
    """Add source admittances (per-unit, one entry per bus, 0 if no source) to the
    network Y-bus diagonal and invert to get the Z-bus."""
    Y = np.array(Y_network, dtype=complex)
    Y[np.diag_indices_from(Y)] += source_admittance
    return np.linalg.inv(Y)


def bolted_three_phase(Z, v_prefault=1.0):
    """Fault current magnitude (per-unit) for a bolted 3-phase fault at every bus."""
    v = np.broadcast_to(v_prefault, (Z.shape[0],)).astype(complex)
    return np.abs(v / np.diag(Z))


def post_fault_voltages(Z, k, v_prefault=1.0):
    """Bus voltages (per-unit, complex) during a bolted fault at bus k."""
    v = np.broadcast_to(v_prefault, (Z.shape[0],)).astype(complex)
    i_f = v[k] / Z[k, k]
    return v - Z[:, k] * i_f


def pu_to_ka(i_pu, base_mva, base_kv):
    """Convert per-unit current to kA: I_base = S_base / (sqrt(3) * V_base)."""
    return i_pu * base_mva / (np.sqrt(3) * base_kv)
