import json

import numpy as np
import pytest

from ezspec import ops, units
from ezspec.pipeline import Pipeline
from ezspec.spectrum import SigmaSource, Spectrum, spectrum

rng = np.random.default_rng(0)


def make(n=500, noise=0.02):
    x = np.linspace(400, 800, n)
    base = 1 + 1e-3 * (x - 400)
    peak = 2 * np.exp(-0.5 * ((x - 600) / 8) ** 2)
    return spectrum(x, base + peak + rng.normal(scale=noise, size=n), x_unit="nm")


def test_spectrum_is_immutable_and_sorted():
    s = spectrum([3, 1, 2], [30, 10, 20])
    assert list(s.x) == [1, 2, 3] and list(s.y) == [10, 20, 30]
    with pytest.raises(ValueError):
        s.x[0] = 5
    with pytest.raises(ValueError):
        Spectrum(np.arange(3.0), np.arange(3.0), sigma=np.array([1.0, 0.0, 1.0]), sigma_source="known")
    assert spectrum([1, 2], [1, 2], sigma=[1, 1]).sigma_source is SigmaSource.KNOWN


def test_crop_exclude_and_validation():
    s = make()
    c = ops.crop(s, xmin=500, xmax=700)
    assert c.x.min() >= 500 and c.x.max() <= 700
    e = ops.exclude(c, ranges=[[590, 610]])
    assert e.exclude.sum() > 0 and not e.fit_mask[np.argmin(abs(e.x - 600))]
    with pytest.raises(TypeError):
        ops.crop(s, xmn=1)
    with pytest.raises(ValueError):
        ops.baseline_asls(s, lam=-1)


def test_unit_conversion_with_jacobian_conserves_area():
    lam = np.linspace(400, 800, 4001)
    f = np.exp(-0.5 * ((lam - 600) / 10) ** 2)
    s = spectrum(lam, f, x_unit="nm")
    e = ops.convert_x(s, to="eV", spectral_density=True)
    assert np.all(np.diff(e.x) > 0)
    area_nm = np.trapezoid(s.y, s.x)
    area_ev = np.trapezoid(e.y, e.x)
    assert area_ev == pytest.approx(area_nm, rel=1e-5)
    r = ops.convert_x(s, to="raman", laser_nm=532.0)
    assert units.convert(r.x, "raman", "nm", 532.0) == pytest.approx(np.sort(lam))
    assert units.HC_EV_NM == pytest.approx(1239.841984, rel=1e-9)


def test_resample_bin_mean_conserves_area_and_propagates_sigma():
    s = spectrum(np.linspace(0, 10, 1001), np.sin(np.linspace(0, 10, 1001)) + 2,
                 sigma=np.full(1001, 0.1))
    r = ops.resample(s, n=101, method="bin_mean")
    assert "interpolated" in r.flags
    assert np.sum(r.y) * (10 / 101) == pytest.approx(np.trapezoid(s.y, s.x), rel=1e-6)
    # averaging ~10 independent points reduces sigma by ~sqrt(10)
    assert np.median(r.sigma) == pytest.approx(0.1 / np.sqrt(10), rel=0.1)


def test_despike_removes_cosmic_ray():
    s = make(noise=0.01)
    y = s.y.copy()
    y[123] += 50
    sp = ops.despike(s.replace(y=y))
    assert abs(sp.y[123] - s.y[123]) < 0.1
    assert sp.meta["despike_replaced"] >= 1


def test_smoothing_display_vs_data():
    s = make()
    d = ops.smooth_moving_average(s, half_window=3)
    assert "smoothed" in d.aux and np.array_equal(d.y, s.y) and "smoothed" not in d.flags
    d = ops.smooth_savgol(s, window=11, polyorder=3, target="data")
    assert "smoothed" in d.flags


def test_noise_estimates():
    s = make(n=20000, noise=0.05)
    e = ops.estimate_noise(s)
    assert e.sigma_source is SigmaSource.ESTIMATED_DERSNR
    assert e.sigma[0] == pytest.approx(0.05, rel=0.05)
    r = ops.estimate_noise(s, method="region", region=[400, 500])
    assert r.sigma[0] == pytest.approx(0.05, rel=0.05)


@pytest.mark.parametrize("op,kw", [
    ("baseline_asls", dict(lam=1e7, p=0.001)),
    ("baseline_arpls", dict(lam=1e7)),
    ("baseline_snip", dict(max_half_window=60)),
    ("baseline_polynomial", dict(order=1, ranges=[[400, 550], [650, 800]])),
    ("baseline_polynomial", dict(order=1, method="imodpoly")),
    ("baseline_anchors", dict(anchors=[[400, None], [550, None], [650, None], [800, None]], window=5)),
    ("baseline_rubberband", dict()),
])
def test_baselines_recover_linear_background(op, kw):
    s = make(n=800, noise=0.005)
    out = getattr(ops, op)(s, **kw)
    truth = 1 + 1e-3 * (s.x - 400)
    away = np.abs(s.x - 600) > 40
    tol = 0.05 if op == "baseline_rubberband" else 0.03
    assert np.max(np.abs(out.aux["baseline"][away] - truth[away])) < tol
    assert "baseline_subtracted" in out.flags
    assert out.meta["baselines"][-1]["method"]


def test_polynomial_variants_match_pybaselines():
    pb = pytest.importorskip("pybaselines")
    s = make(n=600)
    fitter = pb.Baseline(s.x)
    for variant in ("modpoly", "imodpoly"):
        ref, _ = getattr(fitter, variant)(s.y, poly_order=3)
        out = ops.baseline_polynomial(s, order=3, method=variant)
        np.testing.assert_allclose(out.aux["baseline"], ref, rtol=1e-10, atol=1e-10)


def test_masks_keep_whittaker_out_of_peak():
    s = make(n=800, noise=0.005)
    plain = ops.baseline_asls(s, lam=1e4, p=0.5)
    masked = ops.baseline_asls(s, lam=1e4, p=0.5, exclude_ranges=[[560, 640]])
    i = np.argmin(abs(s.x - 600))
    truth = 1 + 1e-3 * 200
    assert abs(masked.aux["baseline"][i] - truth) < 0.02 < abs(plain.aux["baseline"][i] - truth)


def test_pipeline_run_cache_serialisation_and_errors():
    s = make()
    p = Pipeline()
    a = p.add("crop", {"xmin": 450, "xmax": 750})
    b = p.add("baseline_arpls", {"lam": 1e6})
    run = p.run(s)
    assert run.ok and run.final.x.min() >= 450 and "baseline" in run.final.aux
    assert run.raw is s and s.x.min() == 400  # raw untouched
    run2 = p.run(s)
    assert all(r.cached for r in run2.results)
    p.update(b.id, {"lam": 1e7})
    run3 = p.run(s)
    assert run3.results[0].cached and not run3.results[1].cached
    d = json.loads(json.dumps(p.to_dict()))
    q = Pipeline.from_dict(d)
    np.testing.assert_array_equal(q.run(s).final.y, run3.final.y)
    assert q.steps[1].inputs == [a.id]
    p.set_enabled(a.id, False)
    assert p.run(s).final.n == s.n
    p.add("crop", {"xmin": 2000})
    r = p.run(s)
    assert not r.ok and "fewer than 2" in r.results[-1].error
    lines = q.script_lines()
    assert lines[0].startswith("s = ops.crop(s, xmin=450.0")
