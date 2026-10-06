import numpy as np
import pytest

from ezspec import Model, add_peak, spectrum
from ezspec import lineshapes as ls
from ezspec.fit import FitOptions, fit, fit_global, fit_series, split_global


def series(n_spec=8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 100, 400)
    out, centers = [], []
    for i in range(n_spec):
        c = 40 + 1.5 * i
        centers.append(c)
        y = ls.gaussian(x, 30, c, 5) + ls.gaussian(x, 15, 70, 6) + rng.normal(scale=0.03, size=x.size)
        out.append(spectrum(x, y, sigma=np.full(x.size, 0.03), meta={"name": f"T{i}"}))
    return out, np.array(centers)


def test_series_with_start_value_propagation():
    specs, centers = series()
    m = Model()
    add_peak(m, "gaussian", 40, 2, 5)
    add_peak(m, "gaussian", 70, 1, 6)
    res = fit_series(specs, m, propagate=True, index=np.arange(len(specs)) * 10.0)
    v, e = res.parameter("p1_center")
    assert np.all(np.isfinite(e))
    np.testing.assert_allclose(v, centers, atol=5 * e.max())
    h, he = res.parameter("p1.height")
    assert np.all(np.isfinite(h)) and np.all(he > 0)
    rows = res.rows()
    assert rows[3]["name"] == "T3" and rows[3]["index"] == 30.0 and rows[3]["success"]


def test_series_records_failures(tmp_path):
    specs, _ = series(3)
    bad = spectrum([1.0, 2.0], [1.0, 2.0])
    m = Model()
    add_peak(m, "gaussian", 40, 2, 5)
    res = fit_series([specs[0], bad, specs[1]], m)
    assert res.results[1] is None and res.errors[1]
    res.to_csv(tmp_path / "s.csv")
    assert "p1_center_stderr" in (tmp_path / "s.csv").read_text()


def test_global_fit_shared_width_matches_independent_fits_and_reduces_error():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 50, 300)
    specs = []
    for i, (a, c) in enumerate(((10, 20), (6, 25), (8, 30))):
        y = ls.lorentzian(x, a, c, 4.0) + 0.1 + rng.normal(scale=0.01, size=x.size)
        specs.append(spectrum(x, y, sigma=np.full(x.size, 0.01)))
    template = Model()
    add_peak(template, "lorentzian", 25, 1, 3)
    template.add("constant", c=0.0)
    models = []
    for c in (20, 25, 30):
        m = template.copy()
        m.component("p1_").settings["center"].value = c
        models.append(m)
    g = fit_global(specs, models, shared=["p1_fwhm"])
    assert "p1_fwhm" in g.params and "p1_fwhm_d0" not in g.params and "p1_area_d2" in g.params
    assert g.stats.n_varys == 1 + 3 * 3
    assert g.value("p1_fwhm") == pytest.approx(4.0, abs=4 * g.stderr("p1_fwhm"))
    single = [fit(s, m) for s, m in zip(specs, models)]
    se_single = [r.stderr("p1_fwhm") for r in single]
    assert g.stderr("p1_fwhm") < min(se_single)
    # chi2 of the global fit is close to the sum of the independent chi2 (+ cost of the constraint)
    assert g.stats.chi2 >= sum(r.stats.chi2 for r in single) - 1e-6
    part = split_global(g, 1)
    assert len(part["x"]) == 300 and np.allclose(part["y"], specs[1].y)
    fr = {d.component: d.value for d in g.derived if d.name == "area"}
    assert fr["d1.p1"] == pytest.approx(6, rel=0.02)
    assert not any(d.name == "area_fraction" for d in g.derived)   # one peak per spectrum


def test_global_fit_with_range_and_constraint_rewriting():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 50, 300)
    specs = [spectrum(x, ls.gaussian(x, 5, 20, 3) + ls.gaussian(x, 5 * k, 30, 3) + rng.normal(scale=0.01, size=300),
                      sigma=np.full(300, 0.01)) for k in (1.0, 2.0)]
    m = Model()
    add_peak(m, "gaussian", 20, 1, 3)
    add_peak(m, "gaussian", 30, 1, 3)
    m.component("p2_").settings["fwhm"].expr = "p1_fwhm"
    g = fit_global(specs, m, shared=["p1_center"], options=FitOptions(x_range=[5, 45]))
    assert g.params["p2_fwhm_d1"].expr == "p1_fwhm_d1"
    assert g.value("p2_area_d1") / g.value("p2_area_d0") == pytest.approx(2, rel=0.02)
    assert g.stats.n_points == 2 * int(np.sum((x >= 5) & (x <= 45)))
