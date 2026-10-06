import numpy as np
import pytest
from scipy import optimize

from ezspec import Model, add_peak, ops, spectrum
from ezspec import lineshapes as ls
from ezspec.fit import FitError, FitOptions, bootstrap, compare, fit, profile_ci
from ezspec.fit.compare import ComparisonError
from ezspec.models import Formula, FormulaError, add_template, find_peaks

rng = np.random.default_rng(42)


def two_peaks(n=600, sigma=0.05, seed=None):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 100, n)
    y = ls.gaussian(x, 30, 40, 4) + ls.lorentzian(x, 20, 55, 6) + 0.5 + 0.002 * x
    return x, y + r.normal(scale=sigma, size=n), y


def peak_model():
    m = Model()
    add_peak(m, "gaussian", 39, 3, 5)
    add_peak(m, "lorentzian", 56, 2, 5)
    m.add("linear", intercept=0.4, slope=0.0)
    return m


# ------------------------------------------------------------------------------ formula
def test_formula_parsing_and_safety():
    f = Formula("y0 + A*exp(-x/tau)")
    assert f.parameters == ("y0", "A", "tau")
    np.testing.assert_allclose(f(np.array([0.0, 1.0]), y0=1, A=2, tau=1), [3, 1 + 2 / np.e])
    assert Formula("y = a*x^2").parameters == ("a",)
    assert Formula("gauss(x, A, c, w) + b").parameters == ("A", "c", "w", "b")
    for bad in ["__import__('os')", "x.real", "x[0]", "(lambda: 1)()", "open('f')", "a*b", "", "a +"]:
        with pytest.raises(FormulaError):
            Formula(bad)


def test_multiple_independent_variables():
    x1 = rng.uniform(0, 1, 200)
    x2 = rng.uniform(0, 1, 200)
    y = 1.5 + 2 * x1 - 3 * x2 * x1 + rng.normal(scale=0.01, size=200)
    s = spectrum(x1, y, aux={"var:x2": x2})
    m = Model()
    m.add("formula", options={"expression": "a + b*x1 + c*x1*x2", "independent": ["x1", "x2"]})
    r = fit(s, m)
    assert r.value("a") == pytest.approx(1.5, abs=0.01)
    assert r.value("c") == pytest.approx(-3, abs=0.02)


# ------------------------------------------------------------------------------ semantics
def test_known_sigma_absolute_covariance_and_chi2():
    x, y, truth = two_peaks(seed=1)
    s = spectrum(x, y, sigma=np.full_like(x, 0.05))
    r = fit(s, peak_model())
    st = r.stats
    assert st.covariance_mode == "absolute" and st.chi2 is not None
    half = st.redchi_band[1] - 1
    assert abs(st.redchi - 1) < 3 * half
    assert st.ic_k == st.n_varys and "χ² + 2K" in st.ic_form
    assert r.value("p1_area") == pytest.approx(30, abs=4 * r.stderr("p1_area"))
    # scaling: stderr multiplied by sqrt(redchi), flagged with a warning
    r2 = fit(s, peak_model(), FitOptions(covariance="scaled"))
    ratio = r2.stderr("p1_area") / r.stderr("p1_area")
    assert ratio == pytest.approx(np.sqrt(st.redchi), rel=1e-6)
    assert any(w.code == "SCALED_KNOWN_SIGMA" for w in r2.warnings)


def test_unknown_sigma_reports_no_chi2():
    x, y, _ = two_peaks(seed=2)
    r = fit(spectrum(x, y), peak_model())
    st = r.stats
    assert st.chi2 is None and st.redchi is None and st.covariance_mode == "scaled"
    assert st.ic_k == st.n_varys + 1
    assert st.s_res == pytest.approx(0.05, rel=0.1)
    assert "χ²: nicht definiert" in r.report()


def test_wrong_sigma_is_flagged_not_hidden():
    x, y, _ = two_peaks(seed=3, sigma=0.1)
    r = fit(spectrum(x, y, sigma=np.full_like(x, 0.05)), peak_model())
    assert r.stats.redchi > 3 and any(w.code == "REDCHI_HIGH" for w in r.warnings)


@pytest.mark.slow
def test_monte_carlo_coverage_of_error_bars():
    """68.3 % intervals must cover the true value in ~68 % of repetitions."""
    hits = {"p1_area": 0, "p1_center": 0, "p2_fwhm": 0}
    truth = {"p1_area": 30.0, "p1_center": 40.0, "p2_fwhm": 6.0}
    redchis = []
    n_rep = 300
    for k in range(n_rep):
        x, y, _ = two_peaks(n=300, sigma=0.05, seed=1000 + k)
        r = fit(spectrum(x, y, sigma=np.full_like(x, 0.05)), peak_model())
        redchis.append(r.stats.redchi)
        for name in hits:
            if abs(r.value(name) - truth[name]) <= r.stderr(name):
                hits[name] += 1
    for name, h in hits.items():
        frac = h / n_rep
        # binomial SD at p = 0.683, n = 300 is 0.027 -> allow ~3.5 SD
        assert 0.59 < frac < 0.78, f"{name}: coverage {frac:.3f}"
    assert np.mean(redchis) == pytest.approx(1.0, abs=0.03)


def test_constraints_reduce_free_parameters():
    x, y, _ = two_peaks(seed=4)
    m = peak_model()
    m.component("p2_").settings["fwhm"].expr = "p1_fwhm"
    r = fit(spectrum(x, y), m)
    assert r.stats.n_varys == 7
    assert r.value("p2_fwhm") == r.value("p1_fwhm")
    assert r.stderr("p2_fwhm") == pytest.approx(r.stderr("p1_fwhm"), rel=1e-6)


def test_derived_height_matches_analytic_propagation():
    x, y, _ = two_peaks(seed=5)
    r = fit(spectrum(x, y, sigma=np.full_like(x, 0.05)), peak_model())
    d = r.derived_table()["p1"]
    A, w = r.value("p1_area"), r.value("p1_fwhm")
    c = np.sqrt(4 * np.log(2) / np.pi)
    h = c * A / w
    i, j = r.var_names.index("p1_area"), r.var_names.index("p1_fwhm")
    C = r.covariance
    g = np.array([c / w, -c * A / w**2])
    sub = np.array([[C[i, i], C[i, j]], [C[j, i], C[j, j]]])
    assert d["height"][0] == pytest.approx(h, rel=1e-12)
    assert d["height"][1] == pytest.approx(np.sqrt(g @ sub @ g), rel=1e-5)
    fr = r.derived_table()
    assert fr["p1"]["area_fraction"][0] + fr["p2"]["area_fraction"][0] == pytest.approx(1)


def test_voigt_derived_fwhm_is_exact():
    x = np.linspace(-20, 20, 801)
    y = ls.voigt(x, 5, 0.3, 1.2, 0.8) + rng.normal(scale=0.002, size=x.size)
    m = Model()
    add_peak(m, "voigt", 0, 2, 2)
    r = fit(spectrum(x, y), m)
    fw = r.derived_table()["p1"]["fwhm"][0]
    g, l = r.value("p1_fwhm_g"), r.value("p1_fwhm_l")
    assert fw == pytest.approx(ls.voigt_fwhm_exact(g, l), rel=1e-12)
    assert fw == pytest.approx(ls.voigt_fwhm_approx(g, l), rel=3e-4)


def test_poisson_irls_equals_poisson_maximum_likelihood():
    x = np.linspace(0, 10, 200)
    lam = 3 + ls.gaussian(x, 80, 5, 1.5)
    counts = rng.poisson(lam).astype(float)
    m = Model()
    add_peak(m, "gaussian", 5, 15, 2)
    m.add("constant", c=2.0)
    r = fit(spectrum(x, counts), m, FitOptions(weighting="poisson_model"))

    def negll(t):
        f = ls.gaussian(x, t[0], t[1], t[2]) + t[3]
        if np.any(f <= 0):
            return 1e30
        return np.sum(f - counts * np.log(f))
    t0 = [r.value(n) for n in ("p1_area", "p1_center", "p1_fwhm", "bg1_c")]
    ml = optimize.minimize(negll, t0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12,
                                                                        "maxiter": 20000})
    np.testing.assert_allclose(t0, ml.x, rtol=1e-5)
    assert r.stats.covariance_mode == "absolute" and "Poisson" in r.stats.ic_form


def test_flags_and_diagnostics():
    x, y, _ = two_peaks(seed=6)
    s = ops.smooth_savgol(spectrum(x, y), window=11, polyorder=2, target="data")
    r = fit(s, peak_model())
    assert any(w.code == "SMOOTHED_INPUT" for w in r.warnings)
    # wrong model: single Gaussian for two peaks -> runs test
    m = Model()
    add_peak(m, "gaussian", 45, 3, 15)
    r = fit(spectrum(x, y), m)
    assert any(w.code == "RUNS_TEST" for w in r.warnings)
    # parameter driven to its bound
    m = peak_model()
    m.component("p2_").settings["fwhm"].max = 4.0
    m.component("p2_").settings["fwhm"].value = 3.0
    r = fit(spectrum(x, y), m)
    assert r.params["p2_fwhm"].at_bound == "max"
    assert not r.params["p2_fwhm"].reliable_stderr


def test_unidentifiable_parameters_are_reported():
    x = np.linspace(0, 1, 50)
    y = 2 * x + 1 + rng.normal(scale=0.01, size=50)
    m = Model()
    m.add("formula", options={"expression": "a*b*x + c"})   # only a*b is identifiable
    r = fit(spectrum(x, y), m)
    codes = {w.code for w in r.warnings}
    assert codes & {"NO_COVARIANCE", "ILL_CONDITIONED"}


def test_errors():
    x, y, _ = two_peaks()
    with pytest.raises(FitError):
        fit(spectrum(x, y), peak_model(), FitOptions(weighting="sigma"))
    m = Model()
    m.add("formula", options={"expression": "a*x"})
    with pytest.raises(FitError):
        fit(spectrum([1.0], [1.0]), m)


def test_global_method_with_polish():
    x, y, _ = two_peaks(seed=7)
    m = peak_model()
    for c in m.components:
        for k, st in c.settings.items():
            st.min, st.max = (st.value - 10, st.value + 10) if k != "fwhm" else (0.5, 15)
    m.component("bg1_").settings["slope"].min = -1
    m.component("bg1_").settings["slope"].max = 1
    r = fit(spectrum(x, y), m, FitOptions(method="differential_evolution"))
    assert r.method.endswith("+ leastsq") and r.value("p1_center") == pytest.approx(40, abs=0.05)


# ------------------------------------------------------------------------------ uncertainty
def test_profile_ci_equals_linear_se_for_linear_model():
    x = np.linspace(0, 10, 100)
    y = 1 + 0.5 * x + rng.normal(scale=0.2, size=100)
    m = Model()
    m.add("formula", options={"expression": "a + b*x"})
    r = fit(spectrum(x, y, sigma=np.full(100, 0.2)), m)
    ci = profile_ci(r, levels=(0.6827,))
    lo, hi = ci["intervals"]["b"]["0.6827"]
    se = r.stderr("b")
    assert hi - r.value("b") == pytest.approx(se, rel=1e-3)
    assert r.value("b") - lo == pytest.approx(se, rel=1e-3)
    # the bootstrap measures the empirical residual spread, i.e. it corresponds to the
    # *scaled* standard error se * sqrt(redchi)
    se_scaled = se * np.sqrt(r.stats.redchi)
    bs = bootstrap(r, n_samples=800, kind="residual", seed=1)
    assert bs["params"]["b"]["std"] == pytest.approx(se_scaled, rel=0.1)
    bw = bootstrap(r, n_samples=800, kind="wild", seed=2)
    assert bw["params"]["b"]["std"] == pytest.approx(se_scaled, rel=0.1)
    assert "Profil" in r.report() and "Bootstrap" in r.report()


def test_profile_ci_is_asymmetric_for_nonlinear_parameter():
    x = np.linspace(0, 5, 40)
    y = 3 * np.exp(-x / 0.8) + rng.normal(scale=0.05, size=40)
    m = Model()
    m.add("formula", options={"expression": "A*exp(-x/tau)", "defaults": {"A": 2, "tau": 1}})
    r = fit(spectrum(x, y, sigma=np.full(40, 0.05)), m)
    lo, hi = profile_ci(r, names=["tau"], levels=(0.9545,))["intervals"]["tau"]["0.9545"]
    assert lo < r.value("tau") < hi


def test_model_comparison():
    x, y, _ = two_peaks(seed=8)
    s = spectrum(x, y, sigma=np.full_like(x, 0.05))
    good = fit(s, peak_model())
    m = Model()
    add_peak(m, "gaussian", 39, 3, 5)
    add_peak(m, "gaussian", 56, 2, 5)
    m.add("linear", intercept=0.4)
    worse = fit(s, m)
    c = compare({"G+L": good, "G+G": worse})
    assert c["rows"][0]["model"] == "G+L"
    assert sum(r["akaike_weight"] for r in c["rows"]) == pytest.approx(1)
    other = fit(ops.crop(s, xmin=10), peak_model())
    with pytest.raises(ComparisonError):
        compare({"a": good, "b": other})


# ------------------------------------------------------------------------------ library
def test_templates_and_peak_finding():
    x = np.linspace(0, 10, 300)
    y = 2 + 5 * np.exp(-x / 1.7) + rng.normal(scale=0.01, size=300)
    m = Model()
    add_template(m, "Exp. Zerfall (1)")
    r = fit(spectrum(x, y), m)
    assert r.value("tau") == pytest.approx(1.7, rel=0.01)
    assert not r.params["x0"].vary
    xs, ys, _ = two_peaks(seed=9)
    found = find_peaks(xs, ys)
    centers = sorted(p.center for p in found)
    assert len(found) == 2 and abs(centers[0] - 40) < 1 and abs(centers[1] - 55) < 1


def test_serialisation_roundtrip():
    x, y, _ = two_peaks(seed=10)
    m = peak_model()
    m.component("p2_").settings["fwhm"].expr = "p1_fwhm"
    m2 = Model.from_dict(m.to_dict())
    r1 = fit(spectrum(x, y), m)
    r2 = fit(spectrum(x, y), m2)
    assert r1.values == r2.values
    d = r1.to_dict()
    import json
    json.dumps(d)
    assert d["statistics"]["covariance_mode"] == "scaled"


def test_baseline_systematics_reports_spread():
    from ezspec.examples import raman_example
    from ezspec.fit import baseline_systematics
    from ezspec.pipeline import Pipeline
    raw = raman_example(spike=False)
    p = Pipeline()
    p.add("baseline_arpls", {"lam": 1e7})
    m = Model()
    add_peak(m, "lorentzian", 1602, 37, 12)
    opts = FitOptions(x_range=[1500, 1700])
    res = baseline_systematics(raw, p, m, opts)
    assert res["variants"] == [3e6, 1e7, 3e7]
    area = res["table"]["p1_area"]
    assert len(area["values"]) == 3 and area["systematic"] >= 0
    assert any("syst., Baseline" in line for line in res["summary"])


@pytest.mark.slow
def test_simulated_nested_test_calibration():
    from ezspec.fit import simulate_nested_test
    x = np.linspace(0, 100, 300)
    rng_ = np.random.default_rng(11)
    sigma = 0.05
    y = ls.gaussian(x, 30, 40, 5) + rng_.normal(scale=sigma, size=x.size)       # no second peak
    s = spectrum(x, y, sigma=np.full_like(x, sigma))
    m0 = Model()
    add_peak(m0, "gaussian", 40, 2, 5)
    m1 = m0.copy()
    add_peak(m1, "gaussian", 70, 0.05, 5)
    m1.component("p2_").settings["area"].min = 0.0                               # boundary problem
    r0, r1 = fit(s, m0), fit(s, m1)
    res = simulate_nested_test(r0, r1, n_sim=150, seed=2)
    assert res["n"] >= 140 and 0.0 < res["p_value"] <= 1.0
    assert res["p_value"] > 0.01                     # no evidence for a peak that is not there
    # with a real second peak the evidence is overwhelming
    y2 = y + ls.gaussian(x, 1.0, 70, 5)
    s2 = spectrum(x, y2, sigma=np.full_like(x, sigma))
    res2 = simulate_nested_test(fit(s2, m0), fit(s2, m1), n_sim=50, seed=3)
    assert res2["p_value"] < 0.05


def test_poisson_deviance_reported():
    x = np.linspace(0, 10, 200)
    counts = np.random.default_rng(1).poisson(3 + ls.gaussian(x, 80, 5, 1.5)).astype(float)
    m = Model()
    add_peak(m, "gaussian", 5, 15, 2)
    m.add("constant", c=2.0)
    r = fit(spectrum(x, counts), m, FitOptions(weighting="poisson_model"))
    assert r.stats.poisson_deviance / r.stats.dof == pytest.approx(1.0, abs=0.35)
    assert "Poisson-Devianz" in r.report()


def test_mcmc_agrees_with_linear_errors_for_a_well_determined_fit():
    pytest.importorskip("emcee")
    from ezspec.fit import mcmc
    x = np.linspace(0, 100, 300)
    y = ls.gaussian(x, 30, 40, 5) + np.random.default_rng(5).normal(scale=0.05, size=x.size)
    m = Model()
    add_peak(m, "gaussian", 40, 2, 5)
    r = fit(spectrum(x, y, sigma=np.full_like(x, 0.05)), m)
    out = mcmc(r, steps=1500, burn=500, thin=5, seed=1)
    lo, hi = out["params"]["p1_center"]["ci68"]
    assert (hi - lo) / 2 == pytest.approx(r.stderr("p1_center"), rel=0.15)
    assert out["params"]["p1_area"]["median"] == pytest.approx(r.value("p1_area"), abs=r.stderr("p1_area"))
    assert "p1.height" in out["derived"] and 0.1 < out["acceptance"] < 0.9
    assert "MCMC" in r.report()
