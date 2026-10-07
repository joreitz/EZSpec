import numpy as np
import pytest

from ezspec import Model, spectrum
from ezspec.combine import CombineError, combine, diagnose_alignment
from ezspec.fit import FitOptions, fit
from ezspec.spectrum import SigmaSource


def series(seed=0, n=200):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n)
    u = 2.0 + 0.5 * t                     # "voltage"
    i = 0.4 + 0.1 * t                     # "current"
    su, si = 0.01, 0.002
    a = spectrum(t, u + rng.normal(scale=su, size=n), sigma=np.full(n, su), meta={"name": "U"})
    b = spectrum(t, i + rng.normal(scale=si, size=n), sigma=np.full(n, si), meta={"name": "I"})
    return a, b, su, si


def test_quotient_on_identical_grid_with_exact_error_propagation():
    a, b, su, si = series()
    out, rep = combine({"a": a, "b": b}, x_expr="x", y_expr="a/b")
    assert rep["alignment"] == "exact" and out.n == a.n
    q = a.y / b.y
    np.testing.assert_allclose(out.y, q)
    expected = np.abs(q) * np.sqrt((su / a.y) ** 2 + (si / b.y) ** 2)
    np.testing.assert_allclose(out.sigma, expected, rtol=1e-6)
    assert out.sigma_source is SigmaSource.KNOWN and "x_uncertain" not in out.flags


def test_same_input_twice_is_propagated_correctly():
    a, _, su, _ = series()
    out, _ = combine({"a": a}, y_expr="a - a")
    assert out.n == 0 or np.all(out.y == 0)       # exactly zero, sigma zero -> no defined points
    out, _ = combine({"a": a}, y_expr="2*a + a")
    np.testing.assert_allclose(out.sigma, 3 * su, rtol=1e-6)   # fully correlated: 3 sigma, not sqrt(5) sigma


def test_ratio_as_new_x_axis_flags_x_uncertainty_and_effective_variance_fit():
    a, b, su, si = series(n=300)
    rng = np.random.default_rng(3)
    t = a.x
    c_true = 5.0 * (a.y / b.y) + 1.0
    c = spectrum(t, c_true + rng.normal(scale=0.05, size=t.size), sigma=np.full(t.size, 0.05))
    out, rep = combine({"a": a, "b": b, "c": c}, x_expr="a/b", y_expr="c", x_label="U/I", y_label="P")
    assert "x_uncertain" in out.flags and "sigma_x" in out.aux
    assert out.meta["plot_style"] == "scatter" and out.x_label == "U/I"
    assert np.all(np.diff(out.x) >= 0)
    m = Model()
    m.add("formula", options={"expression": "k*x + d", "defaults": {"k": 4, "d": 0}})
    r_ols = fit(out, m)
    assert any(w.code == "X_UNCERTAIN" for w in r_ols.warnings)
    r_ev = fit(out, m, FitOptions(weighting="effective_variance"))
    assert any(w.code == "EFFECTIVE_VARIANCE" for w in r_ev.warnings)
    assert not any(w.code == "X_UNCERTAIN" for w in r_ev.warnings)
    # effective variance enlarges the uncertainty of the slope relative to ignoring sigma_x
    assert r_ev.stderr("k") > r_ols.stderr("k")
    assert r_ev.value("k") == pytest.approx(5.0, abs=4 * r_ev.stderr("k"))


def test_alignment_modes():
    a, b, _, _ = series()
    shifted = spectrum(b.x + 0.2 * np.median(np.diff(b.x)), b.y, sigma=b.sigma)
    assert diagnose_alignment([a, shifted])["mode"] == "index"
    out, rep = combine({"a": a, "b": shifted}, y_expr="a/b")
    assert rep["alignment"] == "index" and out.n == a.n
    other = spectrum(np.linspace(2, 12, 77), np.linspace(1, 2, 77), sigma=np.full(77, 0.01))
    out, rep = combine({"a": a, "b": other}, y_expr="a - b")
    assert rep["alignment"] == "interpolate" and "interpolated" in out.flags
    assert out.x.min() >= 2 and any("Interpolation" in w for w in rep["warnings"])
    with pytest.raises(CombineError):
        combine({"a": a, "b": other}, y_expr="a/b", mode="index")
    with pytest.raises(CombineError):
        combine({"a": a, "b": spectrum([20.0, 21.0], [1.0, 2.0])}, y_expr="a")


def test_division_by_zero_and_bad_names():
    t = np.linspace(-1, 1, 21)
    a = spectrum(t, np.ones(21))
    b = spectrum(t, t)
    out, rep = combine({"a": a, "b": b}, y_expr="a/b")
    assert out.n == 20 and any("undefined" in w for w in rep["warnings"])
    with pytest.raises(CombineError):
        combine({"a": a, "b": b}, y_expr="a/c")
    with pytest.raises(CombineError):
        combine({"x": a}, y_expr="x")
    with pytest.raises(CombineError):
        combine({"exp": a}, y_expr="exp")
