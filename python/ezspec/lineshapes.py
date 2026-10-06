"""Public lineshape functions (area-normalised, widths as FWHM).

These dispatch to the Rust core when available and to the NumPy/SciPy
reference otherwise. All accept array ``x`` and scalar parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._backend import as_f64, core

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _x(x):
    return as_f64(np.atleast_1d(x))


def _ret(x, y):
    return y if np.ndim(x) else float(y[0])


def gaussian(x, area, center, fwhm):
    return _ret(x, core().gaussian(_x(x), float(area), float(center), float(fwhm)))


def lorentzian(x, area, center, fwhm):
    return _ret(x, core().lorentzian(_x(x), float(area), float(center), float(fwhm)))


def voigt(x, area, center, fwhm_g, fwhm_l):
    return _ret(x, core().voigt(_x(x), float(area), float(center), float(fwhm_g), float(fwhm_l)))


def pseudo_voigt(x, area, center, fwhm, eta):
    return _ret(x, core().pseudo_voigt(_x(x), float(area), float(center), float(fwhm), float(eta)))


def tch_pseudo_voigt(x, area, center, fwhm_g, fwhm_l):
    return _ret(x, core().tch_pseudo_voigt(_x(x), float(area), float(center), float(fwhm_g), float(fwhm_l)))


def pearson7(x, area, center, fwhm, m):
    return _ret(x, core().pearson7(_x(x), float(area), float(center), float(fwhm), float(m)))


def emg(x, area, mu, fwhm_g, tau):
    return _ret(x, core().emg(_x(x), float(area), float(mu), float(fwhm_g), float(tau)))


def voigt_fwhm_approx(fwhm_g, fwhm_l):
    """Olivero & Longbothum (1977), ~0.02 % accuracy."""
    return float(core().voigt_fwhm(float(fwhm_g), float(fwhm_l)))


def tch_width_eta(fwhm_g, fwhm_l):
    return tuple(core().tch_width_eta(float(fwhm_g), float(fwhm_l)))


def numeric_fwhm(f, x_peak, scale):
    """FWHM of a unimodal function ``f`` with maximum at ``x_peak`` (bracketing + Brent)."""
    peak = f(x_peak)
    half = 0.5 * peak

    def side(sign):
        step = scale
        for _ in range(200):
            if f(x_peak + sign * step) < half:
                break
            step *= 2.0
        else:
            return np.nan
        return brentq(lambda d: f(x_peak + sign * d) - half, 0.0, step, xtol=1e-14 * max(scale, 1e-300),
                      rtol=1e-13, maxiter=200)

    return side(+1) + side(-1)


def voigt_fwhm_exact(fwhm_g, fwhm_l):
    """Exact FWHM of the Voigt profile (numerical root finding)."""
    g, l = max(fwhm_g, 0.0), max(fwhm_l, 0.0)
    if g == 0 and l == 0:
        return 0.0
    return numeric_fwhm(lambda t: voigt(t, 1.0, 0.0, g, l), 0.0, 0.5 * voigt_fwhm_approx(g, l))
