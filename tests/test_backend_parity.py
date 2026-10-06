"""Rust core vs. Python reference vs. external references (SciPy, pybaselines)."""

import numpy as np
import pytest
from scipy import special

from ezspec import _reference as ref
from ezspec._backend import HAVE_RUST

pytestmark = pytest.mark.skipif(not HAVE_RUST, reason="compiled extension not available")

if HAVE_RUST:
    from ezspec import _core as rs

rng = np.random.default_rng(1234)


def close(a, b, rtol=1e-12, atol=1e-14):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("name,args", [
    ("gaussian", (2.0, 0.3, 1.7)),
    ("lorentzian", (2.0, 0.3, 1.7)),
    ("voigt", (2.0, 0.3, 1.1, 0.6)),
    ("voigt", (2.0, 0.3, 0.0, 0.6)),
    ("voigt", (2.0, 0.3, 1.1, 0.0)),
    ("pseudo_voigt", (2.0, 0.3, 1.1, 0.35)),
    ("tch_pseudo_voigt", (2.0, 0.3, 1.1, 0.6)),
    ("pearson7", (2.0, 0.3, 1.1, 2.5)),
    ("emg", (2.0, 0.3, 1.1, 0.4)),
    ("emg", (2.0, 0.3, 0.05, 3.0)),
])
def test_lineshapes_match_reference(name, args):
    x = np.linspace(-15, 15, 3001)
    close(getattr(rs, name)(x, *args), getattr(ref, name)(x, *args), rtol=1e-12, atol=1e-15)


def test_voigt_matches_scipy_voigt_profile():
    x = np.linspace(-20, 20, 4001)
    sigma, gamma = 0.8, 0.45
    fwhm_g = sigma * 2 * np.sqrt(2 * np.log(2))
    close(rs.voigt(x, 1.0, 0.0, fwhm_g, 2 * gamma), special.voigt_profile(x, sigma, gamma),
          rtol=1e-12, atol=1e-16)


def test_tch_and_olivero_match_reference():
    for g, l in [(1.0, 0.0), (1.0, 0.4), (0.3, 1.2), (0.0, 1.0)]:
        close(rs.tch_width_eta(g, l), ref.tch_width_eta(g, l))
        close(rs.voigt_fwhm(g, l), ref.voigt_fwhm(g, l))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_whittaker_matches_lapack(d):
    y = rng.normal(size=500).cumsum()
    w = rng.uniform(0.0, 1.0, size=500)
    for lam in (1e-2, 1e2, 1e6):
        close(rs.whittaker_smooth(y, w, lam, d), ref.whittaker_smooth(y, w, lam, d), rtol=1e-8, atol=1e-8)


def _spectrum(n=800):
    x = np.linspace(0, 1, n)
    base = 3 + 2 * x + np.sin(3 * x)
    peaks = 8 * np.exp(-((x - 0.3) / 0.01) ** 2) + 5 * np.exp(-((x - 0.65) / 0.03) ** 2)
    return x, base + peaks + rng.normal(scale=0.05, size=n)


def test_asls_arpls_match_reference():
    _, y = _spectrum()
    for fn, kw in [("asls", dict(lam=1e6, p=0.01)), ("arpls", dict(lam=1e5))]:
        r = getattr(rs, fn)(y, **kw)
        p = getattr(ref, fn)(y, **kw)
        close(r[0], p[0], rtol=1e-8, atol=1e-8)
        close(r[1], p[1], rtol=1e-8, atol=1e-8)
        assert len(r[2]) == len(p[2]) and r[3] == p[3]


def test_masked_whittaker_matches_reference():
    x, y = _spectrum()
    fixed = np.full_like(y, np.nan)
    fixed[(x > 0.25) & (x < 0.35)] = 0.0
    fixed[x > 0.95] = 1.0
    for fn in ("asls", "arpls"):
        r = getattr(rs, fn)(y, 1e6, fixed=fixed) if fn == "arpls" else rs.asls(y, 1e6, 0.01, fixed=fixed)
        p = getattr(ref, fn)(y, 1e6, fixed=fixed) if fn == "arpls" else ref.asls(y, 1e6, 0.01, fixed=fixed)
        close(r[0], p[0], rtol=1e-8, atol=1e-8)


def test_snip_rubberband_interp_misc():
    x, y = _spectrum(600)
    for order in (2, 4, 6, 8):
        for dec in (False, True):
            close(rs.snip(y, 25, dec, order), ref.snip(y, 25, dec, order))
    close(rs.rubberband(x, y), ref.rubberband(x, y))
    xa = np.sort(rng.uniform(0, 10, 12))
    ya = rng.normal(size=12)
    xq = np.linspace(-2, 12, 999)
    for ext in ("constant", "linear", "polynomial"):
        close(rs.pchip(xa, ya, xq, ext), ref.pchip(xa, ya, xq, ext), rtol=1e-10, atol=1e-12)
        close(rs.natural_cubic(xa, ya, xq, ext), ref.natural_cubic(xa, ya, xq, ext), rtol=1e-10, atol=1e-12)
        close(rs.linear_interp(xa, ya, xq, ext), ref.linear_interp(xa, ya, xq, ext))
    for mode in ("shrink", "reflect", "nearest"):
        close(rs.moving_average(y, 7, mode), ref.moving_average(y, 7, mode), rtol=1e-12, atol=1e-12)
    assert rs.der_snr(y) == pytest.approx(ref.der_snr(y), rel=1e-14)
    xd, yd = rs.minmax_decimate(x, y, 50)
    xr, yr = ref.minmax_decimate(x, y, 50)
    close(xd, xr)
    close(yd, yr)


def test_errors_are_value_errors():
    with pytest.raises(ValueError):
        rs.whittaker_smooth(np.ones(10), np.zeros(10), 1.0, 2)
    with pytest.raises(ValueError):
        rs.pchip(np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        rs.asls(np.ones(10), 1e3, 1.5)


# ------------------------------------------------------------------ external reference
pybaselines = pytest.importorskip("pybaselines")


@pytest.mark.parametrize("impl", ["rust", "python"])
def test_whittaker_baselines_reproduce_pybaselines(impl):
    mod = rs if impl == "rust" else ref
    x, y = _spectrum(1000)
    fitter = pybaselines.Baseline(x)
    b_ref, p_ref = fitter.asls(y, lam=1e6, p=0.01)
    b, w, hist, _ = mod.asls(y, 1e6, 0.01)
    close(b, b_ref, rtol=1e-8, atol=1e-8)
    close(w, p_ref["weights"])
    close(hist, p_ref["tol_history"], rtol=1e-6, atol=1e-12)
    b_ref, p_ref = fitter.arpls(y, lam=1e5)
    b, w, hist, _ = mod.arpls(y, 1e5)
    close(b, b_ref, rtol=1e-8, atol=1e-8)
    close(w, p_ref["weights"], rtol=1e-6, atol=1e-10)
    assert len(hist) == len(p_ref["tol_history"])


@pytest.mark.parametrize("impl", ["rust", "python"])
@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("decreasing", [False, True])
def test_snip_reproduces_pybaselines(impl, order, decreasing):
    mod = rs if impl == "rust" else ref
    x, y = _spectrum(700)
    b_ref, _ = pybaselines.Baseline(x).snip(y, max_half_window=30, decreasing=decreasing, filter_order=order)
    close(mod.snip(y, 30, decreasing, order), b_ref, rtol=1e-10, atol=1e-10)
