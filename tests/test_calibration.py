import json

import numpy as np
import pytest

from ezspec import Model, ops, spectrum
from ezspec.calibration import Calibration, CalibrationError, calibration_from_fit, parse_assignments
from ezspec.export.figure import curves_for, default_spec, make_resolver, render_figure
from ezspec.fit import fit
from ezspec.models import add_surface
from ezspec.models.formula import variable_name
from ezspec.models.library import surface_expression
from ezspec.pipeline import Pipeline
from ezspec.slices import group_values, slice_curves, slice_variable

C0, CX, CT, X0, T0 = 1530.0, 0.01, 0.1, 50.0, 25.0     # λ = C0 + CX (I - 50) + CT (T - 25)
SIG = 0.002


def calib_data(seed=0, quad=0.0, T_noise=0.0):
    rng = np.random.default_rng(seed)
    I = np.tile(np.linspace(20, 80, 13), 6)
    Tset = np.repeat(np.linspace(15, 40, 6), 13)
    T = Tset + rng.normal(scale=T_noise, size=I.size) if T_noise else Tset
    lam = C0 + CX * (I - X0) + quad * (I - X0) ** 2 + CT * (T - T0) + rng.normal(scale=SIG, size=I.size)
    return spectrum(I, lam, sigma=np.full(I.size, SIG), aux={"var:T": T}, x_label="I", x_unit="mA",
                    y_label="λ", y_unit="nm", meta={"name": "cal"})


def plane_fit(s, kind="plane"):
    m = Model()
    add_surface(m, kind, "T", X0, T0, float(np.median(s.y)))
    return fit(s, m)


def test_surface_expressions_are_centred():
    assert surface_expression("plane", "T", 47.3, -12.4) == "c0 + cx*(x - 47) + cT*(T + 12)"
    assert "cxT*(x - 50)*(T - 25)" in surface_expression("bilinear", "T", 50, 25)
    assert "cTT*(T - 25)**2" in surface_expression("quadratic", "T", 50, 25)


def test_plane_fit_recovers_parameters():
    r = plane_fit(calib_data())
    for name, true in (("c0", C0), ("cx", CX), ("cT", CT)):
        assert abs(r.value(name) - true) < 4 * r.stderr(name)
    assert r.stats.covariance_mode == "absolute"


def test_calibrate_x_slice_and_analytic_sigma():
    s = calib_data()
    r = plane_fit(s)
    cal = calibration_from_fit(r, s, "cal")
    json.dumps(cal)                                          # plain JSON
    x = np.linspace(30, 70, 41)
    d = spectrum(x, np.exp(-0.5 * ((x - 50) / 3) ** 2), x_label="I", x_unit="mA")
    out = ops.calibrate_x(d, calibration=cal, fixed="T=30")
    v = r.values
    expected = v["c0"] + v["cx"] * (x - X0) + v["cT"] * (30 - T0)
    np.testing.assert_allclose(out.x, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out.aux["x_raw"], x)
    assert out.x_label == "λ" and out.x_unit == "nm" and "x_calibrated" in out.flags
    # delta method for a linear model: σ² = g C gᵀ with g = ∂f/∂(c0, cx, cT)
    names = r.var_names
    G = np.column_stack([{"c0": np.ones_like(x), "cx": x - X0, "cT": np.full_like(x, 30 - T0)}[n] for n in names])
    sig = np.sqrt(np.einsum("ij,jk,ik->i", G, r.covariance, G))
    np.testing.assert_allclose(out.aux["sigma_x_cal"], sig, rtol=1e-6)
    assert out.meta["calibration"]["sigma_max"] == pytest.approx(sig.max(), rel=1e-6)
    # uncertainty of the fixed temperature adds (∂λ/∂T σ_T)²
    out2 = ops.calibrate_x(d, calibration=cal, fixed="T=30", fixed_sigma="T=0,05")
    np.testing.assert_allclose(out2.aux["sigma_x_cal"], np.sqrt(sig ** 2 + (v["cT"] * 0.05) ** 2), rtol=1e-6)


def test_fit_on_calibrated_axis_reports_calibration_uncertainty():
    s = calib_data()
    cal = calibration_from_fit(plane_fit(s), s)
    x = np.linspace(30, 70, 200)
    rng = np.random.default_rng(1)
    d = spectrum(x, 1 - 0.4 * np.exp(-0.5 * ((x - 52) / 2) ** 2) + rng.normal(scale=0.005, size=x.size),
                 sigma=np.full(x.size, 0.005), x_unit="mA", x_label="I")
    out = ops.calibrate_x(d, calibration=cal, fixed="T=25")
    m = Model()
    m.add("formula", options={"expression": "b - a*exp(-0.5*((x-c)/w)**2)",
                              "defaults": {"b": 1, "a": 0.4, "c": 1530.02, "w": 0.02}})
    r = fit(out, m)
    codes = [w.code for w in r.warnings]
    assert "X_CALIBRATION" in codes
    assert r.value("c") == pytest.approx(C0 + CX * 2, abs=5 * r.stderr("c") + 1e-3)


def test_per_point_temperature_and_descending_axis():
    s = calib_data()
    cal = calibration_from_fit(plane_fit(s), s)
    x = np.linspace(30, 70, 50)
    T = np.linspace(20, 30, 50)
    d = spectrum(x, x * 0, aux={"var:T": T, "acq_index": np.arange(50.0)}, x_unit="mA")
    out = ops.calibrate_x(d, calibration=cal, fixed="")
    v = cal["values"]
    lam = v["c0"] + v["cx"] * (x - X0) + v["cT"] * (T - T0)
    np.testing.assert_allclose(out.x, lam, atol=1e-9)
    assert out.meta["calibration"]["per_point"] == ["T"]
    with pytest.raises(ValueError, match="Wert für 'T' fehlt"):
        ops.calibrate_x(spectrum(x, x * 0), calibration=cal, fixed="")
    # wavenumber-like calibration with negative slope: output is sorted, acquisition order kept
    neg = spectrum(s.x, 1e7 / s.y, sigma=np.full(s.n, 1e-3), aux={"var:T": s.aux["var:T"]}, x_unit="mA")
    cal_nu = calibration_from_fit(plane_fit(neg), neg)
    out_nu = ops.calibrate_x(d, calibration=cal_nu, fixed="T=25")
    assert np.all(np.diff(out_nu.x) > 0)
    np.testing.assert_array_equal(out_nu.aux["acq_index"], np.arange(50.0)[::-1])


def test_extrapolation_and_hull_warnings():
    s = calib_data()
    cal = calibration_from_fit(plane_fit(s), s)
    x = np.linspace(10, 70, 61)                             # 10 points below I = 20 mA
    out = ops.calibrate_x(spectrum(x, x * 0, x_unit="mA"), calibration=cal, fixed="T=25")
    assert out.meta["calibration"]["n_outside"] == 10
    assert any("Extrapolation" in w for w in out.meta["_step_warnings"])
    # parallelogram-shaped calibration region: inside the bounding box but outside the hull
    I = np.concatenate([np.linspace(20, 50, 7) + dT for dT in (0, 6, 12, 18, 24, 30)])
    T = np.repeat(np.linspace(15, 40, 6), 7)
    sk = spectrum(I, C0 + CX * (I - X0) + CT * (T - T0), sigma=np.full(I.size, SIG), aux={"var:T": T})
    C = Calibration(calibration_from_fit(plane_fit(sk), sk))
    assert C.inside(np.array([35.0]), {"T": 15.0})[0]
    assert not C.inside(np.array([75.0]), {"T": 15.0})[0]   # box: I ≤ 80, but no data at high I and low T
    assert C.inside(np.array([75.0]), {"T": 40.0})[0]


def test_non_monotonic_mapping_is_refused():
    s = calib_data(quad=1e-3)          # vertex at I = 45 mA inside 20…80
    r = plane_fit(s, "quad_x")
    cal = calibration_from_fit(r, s)
    with pytest.raises(ValueError, match="nicht monoton"):
        ops.calibrate_x(spectrum(np.linspace(20, 80, 50), np.zeros(50)), calibration=cal, fixed="T=25")


def test_unit_mismatch_and_bad_input():
    s = calib_data()
    cal = calibration_from_fit(plane_fit(s), s)
    out = ops.calibrate_x(spectrum(np.linspace(0.03, 0.07, 10), np.zeros(10), x_unit="A"), calibration=cal,
                          fixed="T=25")
    assert any("passt nicht zur Kalibrierung" in w for w in out.meta["_step_warnings"])
    with pytest.raises(ValueError, match="unbekannte Variable"):
        ops.calibrate_x(spectrum(np.linspace(30, 70, 10), np.zeros(10)), calibration=cal, fixed="p=1")
    with pytest.raises(CalibrationError):
        ops.calibrate_x(spectrum(np.linspace(30, 70, 10), np.zeros(10)), calibration={}, fixed="")
    assert parse_assignments("T = 25,5; p=-1e-3") == {"T": 25.5, "p": -1e-3}
    with pytest.raises(CalibrationError):
        parse_assignments("T: 25")


def test_pipeline_roundtrip_and_script_literal():
    s = calib_data()
    cal = calibration_from_fit(plane_fit(s), s)
    d = spectrum(np.linspace(30, 70, 30), np.linspace(0, 1, 30), x_unit="mA")
    p = Pipeline()
    p.add("calibrate_x", {"calibration": cal, "fixed": "T=25"})
    a = p.run(d).final
    q = Pipeline.from_dict(json.loads(json.dumps(p.to_dict())))
    np.testing.assert_array_equal(q.run(d).final.x, a.x)
    line = p.script_lines()[0]
    compile(line, "<script>", "exec")                       # inf bounds are written as float('inf')
    ns = {"ops": ops, "s": d}
    exec(line, ns)
    np.testing.assert_array_equal(ns["s"].x, a.x)


def test_slices_grouping_and_figure():
    v = np.repeat([15.0, 20, 25, 30], 10) + np.random.default_rng(0).normal(scale=0.01, size=40)
    g = group_values(v)
    assert len(g) == 4 and all(x.exact for x in g) and [round(x.value) for x in g] == [15, 20, 25, 30]
    cont = group_values(np.linspace(0, 1, 200))
    assert 2 <= len(cont) <= 6 and not cont[0].exact
    s = calib_data(T_noise=0.01)
    assert slice_variable(s) == "T"                          # repeated x grid → slice view
    r = plane_fit(s)
    sl = slice_curves(s, r._internals["model"], r.values)
    assert len(sl) == 6
    np.testing.assert_allclose(sl[0]["yd"][0], r.values["c0"] + r.values["cx"] * (20 - X0)
                               + r.values["cT"] * (sl[0]["value"] - T0), atol=1e-9)
    curves = curves_for(s, None, r)
    keys = [k for k in curves if k.startswith("slice_fit:")]
    assert len(keys) == 6
    spec = default_spec("d", curves, "I (mA)", "λ (nm)")
    fig = render_figure(spec, make_resolver({"d": curves}))
    assert fig is not None


def test_variable_names_from_headers():
    assert variable_name("T / °C") == "T"
    assert variable_name("laser temp. (K)") == "laser_temp"
    assert variable_name("exp") == "exp_v" and variable_name("x") == "x_v"
    assert variable_name("T", {"T"}) == "T2"


def test_surface_fit_on_repeated_x_blocks_is_not_a_multi_sweep():
    from ezspec import Spectrum
    from ezspec.sweeps import is_multivalued
    I = np.tile(np.linspace(20, 80, 13), 6)               # file order: one I ramp per temperature
    T = np.repeat(np.linspace(15, 40, 6), 13)
    lam = C0 + CX * (I - X0) + CT * (T - T0) + np.random.default_rng(2).normal(scale=SIG, size=I.size)
    s = Spectrum(I, lam, np.full(I.size, SIG), "known",
                 aux={"var:T": T, "acq_index": np.arange(I.size, dtype=float)}).sorted()
    assert is_multivalued(s)
    r = plane_fit(s)
    assert "MULTI_SWEEP" not in [w.code for w in r.warnings]
