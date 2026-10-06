"""Spectral x-axis units and conversions.

Physical constants are the exact SI values (CODATA 2018 / SI 2019):
h c / e = 1239.841984... eV nm.

For *spectral densities* (emission, photoluminescence, fluorescence: signal
per unit x) a change of variable requires the Jacobian,
``f(u) = f(v) * |dv/du|``; e.g. ``f(E) = f(lambda) * lambda^2 / (h c)``
(Mooney & Kambhampati 2013). Variances transform with the square of that
factor. For per-point ratio quantities (absorbance, transmittance, epsilon)
only the axis is converted.

Wavelengths are used as given; converting between air and vacuum
wavelengths is the user's responsibility (relevant for line positions).
"""

from __future__ import annotations

import numpy as np

H = 6.62607015e-34          # J s (exact)
C = 299792458.0             # m / s (exact)
E_CHARGE = 1.602176634e-19  # C (exact)
HC_EV_NM = H * C / E_CHARGE * 1e9   # 1239.841984... eV nm

UNITS = {
    "nm": "Wellenlänge λ / nm",
    "cm-1": "Wellenzahl ν̃ / cm⁻¹",
    "eV": "Energie E / eV",
    "raman": "Raman-Verschiebung Δν̃ / cm⁻¹",
}

PLAIN_LABELS = {
    "nm": "Wellenlänge λ (nm)",
    "cm-1": "Wellenzahl ν̃ (cm⁻¹)",
    "eV": "Energie E (eV)",
    "raman": "Raman-Verschiebung (cm⁻¹)",
}

AXIS_LABELS = {
    "nm": r"Wavelength $\lambda$ (nm)",
    "cm-1": r"Wavenumber $\tilde{\nu}$ (cm$^{-1}$)",
    "eV": r"Energy $E$ (eV)",
    "raman": r"Raman shift (cm$^{-1}$)",
}


def _to_nm(x, unit, laser_nm):
    x = np.asarray(x, float)
    if unit == "nm":
        return x
    if unit == "cm-1":
        return 1e7 / x
    if unit == "eV":
        return HC_EV_NM / x
    if unit == "raman":
        _need_laser(laser_nm)
        return 1e7 / (1e7 / laser_nm - x)
    raise ValueError(f"unknown unit {unit!r}")


def _from_nm(lam, unit, laser_nm):
    if unit == "nm":
        return lam
    if unit == "cm-1":
        return 1e7 / lam
    if unit == "eV":
        return HC_EV_NM / lam
    if unit == "raman":
        _need_laser(laser_nm)
        return 1e7 / laser_nm - 1e7 / lam
    raise ValueError(f"unknown unit {unit!r}")


def _need_laser(laser_nm):
    if laser_nm is None or not laser_nm > 0:
        raise ValueError("Raman shift conversion needs the excitation wavelength laser_nm > 0")


def convert(x, src: str, dst: str, laser_nm: float | None = None) -> np.ndarray:
    """Convert axis values between ``nm``, ``cm-1``, ``eV`` and ``raman`` (shift in cm^-1)."""
    if src == dst:
        return np.asarray(x, float).copy()
    return _from_nm(_to_nm(x, src, laser_nm), dst, laser_nm)


def jacobian(x_src, src: str, dst: str, laser_nm: float | None = None) -> np.ndarray:
    """``|d x_src / d x_dst|`` evaluated at the source points.

    A spectral density transforms as ``f_dst = f_src * jacobian``.
    """
    x_src = np.asarray(x_src, float)
    if src == dst:
        return np.ones_like(x_src)
    lam = _to_nm(x_src, src, laser_nm)
    # d(lambda)/d(unit) for each unit, as a function of lambda
    def dlam_du(unit):
        if unit == "nm":
            return np.ones_like(lam)
        if unit == "cm-1":           # lambda = 1e7 / nu  -> dlam/dnu = -lambda^2 / 1e7
            return -lam**2 / 1e7
        if unit == "eV":             # lambda = hc / E    -> dlam/dE = -lambda^2 / hc
            return -lam**2 / HC_EV_NM
        if unit == "raman":          # shift = 1e7/l0 - 1e7/lambda -> dlam/dshift = +lambda^2/1e7
            return lam**2 / 1e7
        raise ValueError(f"unknown unit {unit!r}")
    # dx_src/dx_dst = (dlam/dx_dst) / (dlam/dx_src)
    return np.abs(dlam_du(dst) / dlam_du(src))
