import numpy as np
import pytest
from gridcore.case import load_case
from gridcore.ybus import make_ybus
from gridcore.powerflow import run_powerflow
from gridcore.shortcircuit import zbus, bolted_three_phase, post_fault_voltages, pu_to_ka
from gridcore.estimation import wls, estimate_with_bad_data
import validate_sc


@pytest.fixture(scope="module")
def case14():
    net, ppc = load_case(14)
    return net, ppc, run_powerflow(ppc)


def _max_errors(net, r):
    idx = net._pd2ppc_lookups["bus"][net.res_bus.index.values]
    vm = np.abs(r["V"][idx]); va = np.rad2deg(np.angle(r["V"][idx]))
    return (np.max(np.abs(vm - net.res_bus.vm_pu.values)),
            np.max(np.abs(va - net.res_bus.va_degree.values)))


# ---------- Y-bus ----------
def test_ybus_symmetric_without_phase_shifters(case14):
    _, ppc, r = case14
    assert np.allclose(r["Y"], r["Y"].T)


def test_ybus_rows_sum_to_shunts_when_no_taps():
    # two-bus line: each row of Y sums to that bus's shunt admittance (here only line charging)
    bus = np.zeros((2, 13)); bus[:, 0] = [0, 1]
    branch = np.zeros((1, 13)); branch[0, [0, 1, 2, 3, 4, 10]] = [0, 1, 0.01, 0.1, 0.02, 1]
    Y = make_ybus(100.0, bus, branch)
    assert np.allclose(Y.sum(axis=1), 1j * 0.02 / 2)


# ---------- power flow ----------
@pytest.mark.parametrize("n", [14, 30, 118, 300])
def test_powerflow_matches_pandapower(n):
    net, ppc = load_case(n)
    r = run_powerflow(ppc)
    assert r["converged"]
    dvm, dva = _max_errors(net, r)
    assert dvm < 1e-9 and dva < 1e-7


def test_newton_raphson_converges_quickly(case14):
    assert case14[2]["iterations"] <= 6


def test_power_balance(case14):
    _, ppc, r = case14
    S = r["V"] * np.conj(r["Y"] @ r["V"])
    pq = r["pq"]
    load = (ppc["bus"][pq, 2] + 1j * ppc["bus"][pq, 3]) / ppc["baseMVA"]
    gen_at_pq = np.zeros(len(pq), dtype=complex)  # IEEE 14 has no generators at PQ buses
    assert np.allclose(S[pq], gen_at_pq - load, atol=1e-9)


# ---------- short circuit ----------
def test_short_circuit_matches_pandapower_iec60909():
    ref = validate_sc.reference()
    mine, _ = validate_sc.gridcore_sc()
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_faulted_bus_voltage_is_zero():
    _, Z = validate_sc.gridcore_sc()
    for k in range(Z.shape[0]):
        assert abs(post_fault_voltages(Z, k, 1.1)[k]) < 1e-12


def test_fault_current_is_one_over_thevenin_impedance():
    Y = np.array([[3 - 9j, -1 + 3j], [-1 + 3j, 1.5 - 4.5j]])
    Z = zbus(Y, np.zeros(2))
    assert np.allclose(bolted_three_phase(Z, 1.0), 1 / np.abs(np.diag(np.linalg.inv(Y))))


def test_pu_to_ka():
    assert pu_to_ka(1.0, 100.0, 20.0) == pytest.approx(100 / (np.sqrt(3) * 20))


# ---------- state estimation ----------
from gridcore.meters import scada_measurements


def _se_inputs(ppc, r, seed):
    meas, z, s, br = scada_measurements(r, ppc["branch"], np.random.default_rng(seed))
    ref = int(r["ref"][0])
    return meas, z, s, br, ref, np.angle(r["V"][ref])


def test_wls_recovers_true_state(case14):
    _, ppc, r = case14
    meas, z, s, br, ref, va_ref = _se_inputs(ppc, r, 1)
    res = wls(r["Y"], z, s, ref, meas, br, va_ref)
    nb = len(r["V"])
    assert np.max(np.abs(res["x"][nb - 1:] - np.abs(r["V"]))) < 0.01


def test_clean_data_passes_chi_squared(case14):
    _, ppc, r = case14
    meas, z, s, br, ref, va_ref = _se_inputs(ppc, r, 2)
    res = estimate_with_bad_data(r["Y"], z, s, ref, meas, br, va_ref)
    assert res["removed"] == []


@pytest.mark.parametrize("bad", [("P", 8), ("Q", 4), ("Pf", 6), ("V", 10)])
def test_gross_error_is_detected_and_removed(case14, bad):
    _, ppc, r = case14
    meas, z, s, br, ref, va_ref = _se_inputs(ppc, r, 3)
    i = meas.index(bad); z[i] += 0.5 if bad[0] != "V" else 0.1
    res = estimate_with_bad_data(r["Y"], z, s, ref, meas, br, va_ref)
    assert res["removed"] == [i]
    assert res["J"] <= res["chi2_threshold"]


def test_estimator_keeps_nonzero_reference_angle():
    # IEEE 118's slack bus sits at 30 degrees; estimated angles must stay referenced to it
    net, ppc = load_case(118)
    r = run_powerflow(ppc)
    meas, z, s, br, ref, va_ref = _se_inputs(ppc, r, 4)
    assert abs(np.rad2deg(va_ref)) > 1
    res = wls(r["Y"], z, s, ref, meas, br, va_ref)
    nb = len(r["V"])
    assert np.rad2deg(np.max(np.abs(res["x"][:nb - 1] - np.delete(np.angle(r["V"]), ref)))) < 1.0
