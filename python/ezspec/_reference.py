"""Pure NumPy/SciPy reference implementations of the Rust core.

Every function here has the same signature and semantics as the function of
the same name in the compiled extension ``ezspec._core``. These versions are
the *reference*: the test-suite checks the Rust core against them, and they
are used automatically when the extension is not available.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, special
from scipy.interpolate import CubicSpline, PchipInterpolator

__version__ = "reference"

TINY = 1.0e-15
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_EPS = np.finfo(float).eps


def _w(width):
    return width if width > TINY else TINY


# --------------------------------------------------------------------------- lineshapes
def gaussian(x, area, center, fwhm):
    sigma = _w(fwhm) * FWHM_TO_SIGMA
    t = (np.asarray(x, float) - center) / sigma
    return area * np.exp(-0.5 * t * t) / (sigma * _SQRT_2PI)


def lorentzian(x, area, center, fwhm):
    gamma = 0.5 * _w(fwhm)
    dx = np.asarray(x, float) - center
    return area * gamma / (np.pi * (dx * dx + gamma * gamma))


def voigt(x, area, center, fwhm_g, fwhm_l):
    if not fwhm_g > TINY:
        return lorentzian(x, area, center, fwhm_l)
    sigma = fwhm_g * FWHM_TO_SIGMA
    gamma = 0.5 * fwhm_l if fwhm_l > 0 else 0.0
    z = (np.asarray(x, float) - center + 1j * gamma) / (sigma * np.sqrt(2.0))
    return area * special.wofz(z).real / (sigma * _SQRT_2PI)


def voigt_fwhm(fwhm_g, fwhm_l):
    """Olivero & Longbothum (1977), ~0.02 % accuracy."""
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def pseudo_voigt(x, area, center, fwhm, eta):
    return eta * lorentzian(x, area, center, fwhm) + (1.0 - eta) * gaussian(x, area, center, fwhm)


def tch_width_eta(fwhm_g, fwhm_l):
    g = max(fwhm_g, 0.0)
    l = max(fwhm_l, 0.0)
    s = (g**5 + 2.69269 * g**4 * l + 2.42843 * g**3 * l**2 + 4.47163 * g**2 * l**3
         + 0.07842 * g * l**4 + l**5)
    width = s**0.2
    if not width > 0:
        return 0.0, 0.0
    q = l / width
    return width, 1.36603 * q - 0.47719 * q**2 + 0.11116 * q**3


def tch_pseudo_voigt(x, area, center, fwhm_g, fwhm_l):
    w, eta = tch_width_eta(fwhm_g, fwhm_l)
    return pseudo_voigt(x, area, center, w, eta)


def pearson7(x, area, center, fwhm, m):
    x = np.asarray(x, float)
    if not m > 0.5:
        return np.full_like(x, np.nan)
    a = _w(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / m) - 1.0))
    norm = np.exp(special.gammaln(m) - special.gammaln(m - 0.5)) / (np.sqrt(np.pi) * a)
    t = (x - center) / a
    return area * norm * (1.0 + t * t) ** (-m)


def emg(x, area, mu, fwhm_g, tau):
    x = np.asarray(x, float)
    sigma = _w(fwhm_g) * FWHM_TO_SIGMA
    if not tau > TINY:
        return gaussian(x, area, mu, fwhm_g)
    u = (x - mu) / sigma
    b = (sigma / tau - u) / np.sqrt(2.0)
    pref = area / (2.0 * tau)
    out = np.empty_like(x)
    pos = b >= 0
    out[pos] = pref * np.exp(-0.5 * u[pos] ** 2) * special.erfcx(b[pos])
    neg = ~pos
    a = 0.5 * (sigma / tau) ** 2 - (x[neg] - mu) / tau
    out[neg] = pref * np.exp(a) * special.erfc(b[neg])
    return out


# --------------------------------------------------------------------------- Whittaker
def _penalty_bands(n, d):
    """Lower band storage (LAPACK convention) of D^T D."""
    c = np.array([1.0])
    for _ in range(d):
        c = np.concatenate([[0.0], c]) - np.concatenate([c, [0.0]])
    ab = np.zeros((d + 1, n))
    for row in range(n - d):
        for a in range(d + 1):
            for b in range(a + 1):
                ab[a - b, row + b] += c[a] * c[b]
    return ab


class _System:
    def __init__(self, n, lam, d):
        if not (np.isfinite(lam) and lam >= 0):
            raise ValueError(f"invalid parameter: lam must be finite and >= 0, got {lam}")
        if d < 1:
            raise ValueError("invalid parameter: diff_order must be >= 1")
        if n <= d:
            raise ValueError(f"too few points: need at least {d + 1}, got {n}")
        self.d = d
        self.pen = lam * _penalty_bands(n, d)

    def solve(self, y, w):
        if np.count_nonzero(w > 0) < self.d:
            raise ValueError("linear system is not positive definite (row 0); "
                             "too few points with non-zero weight for this difference order?")
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("invalid parameter: weights must be finite and >= 0")
        ab = self.pen.copy()
        ab[0] += w
        return linalg.solveh_banded(ab, w * y, lower=True, check_finite=False)


def whittaker_smooth(y, weights, lam, diff_order):
    y = np.asarray(y, float)
    return _System(len(y), lam, diff_order).solve(y, np.asarray(weights, float))


def _rel_diff(old, new):
    return np.linalg.norm(new - old) / max(np.linalg.norm(old), _EPS)


def _apply_fixed(w, fixed):
    if fixed is not None:
        m = ~np.isnan(fixed)
        w[m] = fixed[m]
    return w


def _init_weights(n, weights, fixed):
    w = np.ones(n) if weights is None else np.array(weights, float)
    if fixed is not None:
        fixed = np.asarray(fixed, float)
    return _apply_fixed(w, fixed), fixed


def asls(y, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None, fixed=None):
    if not 0 < p < 1:
        raise ValueError("invalid parameter: p must be between 0 and 1")
    y = np.asarray(y, float)
    system = _System(len(y), lam, diff_order)
    w, fixed = _init_weights(len(y), weights, fixed)
    hist, converged, baseline = [], False, None
    for _ in range(max_iter + 1):
        baseline = system.solve(y, w)
        new_w = _apply_fixed(np.where(y > baseline, p, 1.0 - p), fixed)
        diff = _rel_diff(w, new_w)
        hist.append(diff)
        if diff < tol:
            converged = True
            break
        w = new_w
    return baseline, w, np.array(hist), converged


def arpls(y, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None, fixed=None):
    y = np.asarray(y, float)
    system = _System(len(y), lam, diff_order)
    w, fixed = _init_weights(len(y), weights, fixed)
    free = np.ones(len(y), bool) if fixed is None else np.isnan(fixed)
    hist, converged, baseline = [], False, None
    for _ in range(max_iter + 1):
        baseline = system.solve(y, w)
        resid = y - baseline
        neg = resid[free & (resid < 0)]
        if neg.size < 2:
            break
        std = neg.std(ddof=1)
        if std == 0:
            std = _EPS
        new_w = special.expit(-(2.0 / std) * (resid - (2.0 * std - neg.mean())))
        new_w = _apply_fixed(new_w, fixed)
        diff = _rel_diff(w, new_w)
        hist.append(diff)
        if diff < tol:
            converged = True
            break
        w = new_w
    return baseline, w, np.array(hist), converged


# --------------------------------------------------------------------------- SNIP / rubberband
def pad_extrapolate(y, pad):
    y = np.asarray(y, float)
    n = len(y)
    if pad == 0:
        return y.copy()
    w = min(pad, n)
    if w == 1:
        return np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    xl = np.arange(pad, pad + w, dtype=float)
    pl = np.polyfit(xl, y[:w], 1)
    xr = np.arange(pad + n - w, pad + n, dtype=float)
    pr = np.polyfit(xr, y[-w:], 1)
    left = np.polyval(pl, np.arange(pad, dtype=float))
    right = np.polyval(pr, np.arange(pad + n, 2 * pad + n, dtype=float))
    return np.concatenate([left, y, right])


def snip(y, max_half_window, decreasing=False, filter_order=2):
    if filter_order not in (2, 4, 6, 8):
        raise ValueError("invalid parameter: filter_order must be 2, 4, 6 or 8")
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        raise ValueError(f"too few points: need at least 3, got {n}")
    hw = int(min(max(max_half_window, 1), (n - 1) // 2))
    b = pad_extrapolate(y, hw)
    m = len(b)
    rng = range(hw, 0, -1) if decreasing else range(1, hw + 1)
    for i in rng:
        j = np.arange(i, m - i)

        def pair(k):
            return b[j - k] + b[j + k]

        f = 0.5 * pair(i)
        if filter_order > 2:
            f = np.maximum(f, (-pair(i) + 4 * pair(i // 2)) / 6)
        if filter_order > 4:
            f = np.maximum(f, (pair(i) - 6 * pair(2 * i // 3) + 15 * pair(i // 3)) / 20)
        if filter_order > 6:
            f = np.maximum(f, (-pair(i) + 8 * pair(3 * i // 4) - 28 * pair(i // 2)
                               + 56 * pair(i // 4)) / 70)
        b[j] = np.minimum(b[j], f)
    return b[hw:hw + n].copy()


def _check_xy(x, y, needed=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: expected {len(x)}, got {len(y)}")
    if len(x) < needed:
        raise ValueError(f"too few points: need at least {needed}, got {len(x)}")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x values must be strictly increasing")
    return x, y


def rubberband(x, y):
    x, y = _check_xy(x, y)
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])


# --------------------------------------------------------------------------- interpolation
def _extrapolate(x, y, xq, out, ext, d0, d1):
    lo = xq < x[0]
    hi = xq > x[-1]
    if ext == "constant":
        out[lo] = y[0]
        out[hi] = y[-1]
    elif ext == "linear":
        out[lo] = y[0] + d0 * (xq[lo] - x[0])
        out[hi] = y[-1] + d1 * (xq[hi] - x[-1])
    elif ext not in ("polynomial", "extrapolate"):
        raise ValueError(f"invalid parameter: unknown extrapolation '{ext}'")
    return out


def pchip(x, y, xq, extrapolation="constant"):
    x, y = _check_xy(x, y)
    xq = np.asarray(xq, float)
    p = PchipInterpolator(x, y, extrapolate=True)
    dp = p.derivative()
    return _extrapolate(x, y, xq, p(xq), extrapolation, dp(x[0]), dp(x[-1]))


def natural_cubic(x, y, xq, extrapolation="constant"):
    x, y = _check_xy(x, y)
    xq = np.asarray(xq, float)
    s = CubicSpline(x, y, bc_type="natural", extrapolate=True)
    ds = s.derivative()
    return _extrapolate(x, y, xq, s(xq), extrapolation, ds(x[0]), ds(x[-1]))


def linear_interp(x, y, xq, extrapolation="constant"):
    x, y = _check_xy(x, y)
    xq = np.asarray(xq, float)
    out = np.interp(xq, x, y)
    s0 = (y[1] - y[0]) / (x[1] - x[0])
    s1 = (y[-1] - y[-2]) / (x[-1] - x[-2])
    if extrapolation == "constant":
        return out
    return _extrapolate(x, y, xq, out, "linear", s0, s1)


# --------------------------------------------------------------------------- misc
def moving_average(y, half_window, mode="shrink"):
    y = np.asarray(y, float)
    n = len(y)
    k = int(half_window)
    if n == 0 or k == 0:
        return y.copy()
    if mode == "shrink":
        out = np.empty(n)
        c = np.concatenate([[0.0], np.cumsum(y)])
        for i in range(n):
            h = min(k, i, n - 1 - i)
            out[i] = (c[i + h + 1] - c[i - h]) / (2 * h + 1)
        return out
    if mode == "reflect":
        idx = np.arange(-k, n + k)
        period = 2 * (n - 1) if n > 1 else 1
        idx = np.mod(idx, period)
        idx = np.where(idx >= n, period - idx, idx)
        padded = y[idx]
    elif mode == "nearest":
        padded = np.pad(y, k, mode="edge")
    else:
        raise ValueError(f"invalid parameter: unknown edge mode '{mode}'")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(padded, kernel, mode="valid")


def der_snr(y):
    y = np.asarray(y, float)
    n = len(y)
    if n < 5:
        return float("nan")
    d = np.abs(2.0 * y[2:n - 2] - y[0:n - 4] - y[4:n])
    d = d[~np.isnan(d)]
    return 0.6052697 * float(np.median(d)) if d.size else float("nan")


def minmax_decimate(x, y, n_bins):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    if n_bins == 0 or n <= 2 * n_bins:
        return x[:n].copy(), y[:n].copy()
    xo, yo = [], []
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = max((b + 1) * n // n_bins, lo + 1)
        seg = y[lo:hi]
        imin = lo + int(np.argmin(seg))
        imax = lo + int(np.argmax(seg))
        first, second = sorted((imin, imax))
        xo.append(x[first])
        yo.append(y[first])
        if second != first:
            xo.append(x[second])
            yo.append(y[second])
    return np.array(xo), np.array(yo)
