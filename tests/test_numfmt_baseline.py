import csv

import numpy as np
import pytest

from ezspec import Model, add_peak, numfmt, ops
from ezspec.baseline_info import baseline_short, baseline_statement
from ezspec.examples import raman_example
from ezspec.export.figure import BASELINE_ITEM, dataset_curves, dataset_stages, param_box_lines, stage_spec
from ezspec.export.results import write_all
from ezspec.fit import fit
from ezspec.numfmt import fmt_num, fmt_value


@pytest.fixture(autouse=True)
def default_precision():
    numfmt.set_precision(2, 6)
    yield
    numfmt.set_precision(2, 6)


@pytest.mark.parametrize("v, e, n, expected", [
    (1001.03456, 0.0313, 2, "1001.035 ± 0.031"),
    (1001.03456, 0.0313, 4, "1001.03456 ± 0.03130"),
    (0.123456, 0.00961, 1, "0.12 ± 0.01"),            # 0.0096 → 0.01: carry into the next decade
    (12345.6, 321.0, 2, "12350 ± 320"),
    (-0.5, 0.25, 1, "-0.5 ± 0.3"),                     # half-up, as usual for uncertainties
    (1.23456e8, 3.2e5, 2, "(1.2346 ± 0.0032)e+08"),
    (2.3e-6, 1.1e-7, 2, "(2.30 ± 0.11)e-06"),
    (1.5, 0.123456789, 0, "1.5 ± 0.123456789"),        # no rounding
])
def test_gum_rounding(v, e, n, expected):
    assert fmt_value(v, e, n) == expected


def test_global_precision_setting():
    assert fmt_value(1001.03456, 0.0313) == "1001.035 ± 0.031"
    numfmt.set_precision(unc_digits=3, digits=10)
    assert fmt_value(1001.03456, 0.0313) == "1001.0346 ± 0.0313"
    assert fmt_num(1001.0345678) == "1001.034568"
    assert fmt_value(5.0, None) == "5"
    with pytest.raises(ValueError):
        numfmt.set_precision(unc_digits=-1)


def _processed():
    s = ops.baseline_polynomial(raman_example(), order=2, ranges=[[400, 950], [1700, 1800]])
    return s, ops.baseline_arpls(s, lam=1e7, exclude_ranges=[[990, 1010]])


def test_baseline_recipes_are_complete_and_reproduce_the_baseline():
    poly, s = _processed()
    rec_poly, rec_arpls = s.meta["baselines"]
    lo, hi = rec_poly["domain"]
    t = (2 * poly.x - (lo + hi)) / (hi - lo)
    b = sum(c * t ** k for k, c in enumerate(rec_poly["coefficients"]))
    np.testing.assert_allclose(b, poly.aux["baseline"], rtol=0, atol=1e-9)
    assert rec_poly["ranges"] == [[400, 950], [1700, 1800]] and len(rec_poly["coefficient_stderr"]) == 3
    assert rec_arpls["lam"] == 1e7 and rec_arpls["diff_order"] == 2 and rec_arpls["converged"]
    assert rec_arpls["exclude_ranges"] == [[990, 1010]] and rec_arpls["x_unit"] == "cm⁻¹"
    assert rec_arpls["cutoff_x"] == pytest.approx(rec_arpls["cutoff_points"] * np.median(np.diff(s.x)))


def test_polynomial_baseline_standard_errors_match_monte_carlo():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 200)
    truth = 3.0 + 0.4 * x - 0.02 * x ** 2
    est, se = [], []
    for _ in range(400):
        from ezspec import spectrum
        s = spectrum(x, truth + rng.normal(scale=0.5, size=x.size))
        rec = ops.baseline_polynomial(s, order=2, ranges=[[0, 3], [7, 10]]).meta["baselines"][0]
        est.append(rec["coefficients"])
        se.append(rec["coefficient_stderr"])
    sd_mc = np.std(est, axis=0, ddof=1)
    np.testing.assert_allclose(np.mean(se, axis=0), sd_mc, rtol=0.1)


def test_report_results_and_exports_state_the_baseline(tmp_path):
    _, s = _processed()
    m = Model()
    add_peak(m, "lorentzian", 1001, 90, 8)
    r = fit(s, m, {"x_range": [950, 1060]})
    rep = r.report()
    assert "Baseline\n" in rep and "[1] Baseline: polynomial, order 2 (fit)" in rep
    assert "[2] Baseline: arPLS (Whittaker), λ = 1.00e+07" in rep and "excluded ranges [990, 1010] cm⁻¹" in rep
    assert "not included in the parameter errors" in rep
    files = write_all(r, tmp_path / "res")
    rows = dict(csv.reader(open([f for f in files if f.name.endswith("_statistics.csv")][0])))
    assert rows["baseline"].startswith("polynomial, order 2") and "arPLS" in rows["baseline"] and "baseline_2" in rows
    with open([f for f in files if f.name.endswith("_curves.csv")][0]) as fh:
        data = list(csv.DictReader(fh))
    y = np.array([float(d["y"]) for d in data])
    b = np.array([float(d["baseline_subtracted"]) for d in data])
    raw = raman_example()
    keep = (raw.x >= 950) & (raw.x <= 1060)
    np.testing.assert_allclose(y + b, raw.y[keep], rtol=0, atol=1e-9)      # data before baseline removal
    # no baseline: the statement says so explicitly
    r0 = fit(raman_example(), m, {"x_range": [950, 1060]})
    assert "No baseline was subtracted before the fit." in r0.report()
    assert "The model contains no background term." in r0.report()
    m.add("linear")
    assert "Background fitted within the model" in "\n".join(baseline_statement([], m.to_dict()))
    assert baseline_short([], None) == "Baseline: none (no baseline subtracted, no background in the model)"


def test_figures_state_the_baseline():
    from ezspec.pipeline import Pipeline
    p = Pipeline()
    sid = p.add("baseline_arpls", {"lam": 1e7}).id
    run = p.run(raman_example())
    m = Model()
    add_peak(m, "lorentzian", 1001, 90, 8)
    r = fit(run.final, m, {"x_range": [950, 1060]})
    curves = dataset_curves(run.raw, run, r)
    stages = dataset_stages(run.raw, run)
    res_spec = stage_spec("d", curves, stages[0])
    assert res_spec["panels"][0]["param_box"]["lines"][0] == "Baseline: arPLS (Whittaker), λ = 1.00e+07 (subtracted)"
    step_spec = stage_spec("d", curves, next(st for st in stages if st["id"] == sid))
    assert "λ = 1.00e+07" in step_spec["panels"][0]["traces"][-1]["label"]
    lines = param_box_lines(r, [BASELINE_ITEM, "p1_center"], digits=4)
    assert "arPLS" in lines[0] and lines[1].startswith("p1_center = ") and lines[1].count("±") == 1
