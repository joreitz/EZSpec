"""Map the x axis through a fitted calibration (e.g. laser current → wavelength)."""

from __future__ import annotations

import numpy as np

from ..calibration import Calibration, parse_assignments
from ..spectrum import Spectrum
from .basic import Y_DIFFERENCE_AUX, Y_LEVEL_AUX
from .registry import ParamSpec as P
from .registry import operation, warn


@operation("calibrate_x", 1, "x kalibrieren (Fit als Kalibrierung)", "Einheiten",
           params=[P("calibration", "dict", {}, "Kalibrierung",
                     help="aus einem Fit übernommen (Analyse → Fit als x-Kalibrierung anwenden)"),
                   P("fixed", "str", "", "feste Variablen",
                     help="z. B. T=25 – leer: Spalte var:<Name> der Daten (Wert je Punkt)"),
                   P("fixed_sigma", "str", "", "Unsicherheit fester Werte",
                     help="z. B. T=0.02 – geht als systematischer Beitrag in σ_x,cal ein"),
                   P("x_label", "str", "", "neuer Achsentitel"),
                   P("x_unit", "str", "", "neue Einheit"),
                   P("spectral_density", "bool", False, "y ist Dichte pro x-Einheit (Jacobi-Faktor)")],
           flags=("x_calibrated",))
def calibrate_x(s: Spectrum, calibration, fixed, fixed_sigma, x_label, x_unit, spectral_density) -> Spectrum:
    """Replace x by the calibration evaluated at x with the other variables fixed
    (a slice of the fitted surface). The systematic calibration uncertainty
    σ_x,cal (delta method from the fit covariance, plus the uncertainty of the
    fixed values) is stored in aux 'sigma_x_cal' and reported by fits, but not
    used as a random per-point σ."""
    cal = Calibration(calibration)
    fixed_vals = parse_assignments(fixed)
    fixed_sig = parse_assignments(fixed_sigma)
    unknown = (set(fixed_vals) | set(fixed_sig)) - set(cal.variables)
    if unknown:
        raise ValueError(f"unbekannte Variable(n) {', '.join(sorted(unknown))} – die Kalibrierung hat "
                         f"{', '.join(cal.variables) or 'keine weiteren Variablen'}")
    values, per_point = {}, []
    for v in cal.variables:
        if v in fixed_vals:
            values[v] = fixed_vals[v]
        elif f"var:{v}" in s.aux:
            values[v] = np.asarray(s.aux[f"var:{v}"], float)
            per_point.append(v)
        else:
            raise ValueError(f"Wert für {v!r} fehlt: bei 'feste Variablen' z. B. {v}=… eintragen "
                             f"oder die Spalte {v} mit importieren")
    xn = cal.evaluate(s.x, values)
    if not np.all(np.isfinite(xn)):
        raise ValueError("Kalibrierung liefert für einige Punkte keinen endlichen Wert")
    k = cal.slope(s.x, values)
    if np.any(k > 0) and np.any(k < 0):
        raise ValueError("Kalibrierung ist im x-Bereich der Daten nicht monoton (df/dx wechselt das Vorzeichen) "
                         "– die neue Achse wäre nicht eindeutig")
    if np.any(k == 0):
        raise ValueError("df/dx = 0 im Datenbereich – Abbildung nicht umkehrbar")

    sig_cal = cal.sigma(s.x, values)
    var_cal = np.zeros(s.n) if sig_cal is None else sig_cal ** 2
    for v, sv in fixed_sig.items():
        var_cal = var_cal + (cal.partial(v, s.x, values) * sv) ** 2
    sig_cal = np.sqrt(var_cal)

    J = 1.0 / np.abs(k) if spectral_density else np.ones(s.n)       # |dx_old / dx_new|
    aux = {key: (a * J if key in Y_LEVEL_AUX + Y_DIFFERENCE_AUX else a) for key, a in s.aux.items()}
    if "sigma_x" in aux:             # random x uncertainty is mapped through the local slope
        aux["sigma_x"] = np.abs(k) * aux["sigma_x"]
    aux["x_raw"] = np.asarray(s.x, float)
    aux["sigma_x_cal"] = sig_cal
    inside = cal.inside(s.x, values)
    n_out = int((~inside).sum())
    meta = dict(s.meta)
    meta["calibration"] = {
        "source": cal.d.get("source", ""), "expression": cal.d.get("expression", ""),
        "fixed": {key: float(v) for key, v in fixed_vals.items()}, "per_point": per_point,
        "fixed_sigma": fixed_sig,
        "sigma_median": float(np.median(sig_cal)), "sigma_max": float(np.max(sig_cal)),
        "rms_residual": cal.d.get("rms_residual"), "n_outside": n_out,
        "covariance_mode": cal.d.get("covariance_mode"),
    }
    out = s.replace(x=xn, y=s.y * J, sigma=None if s.sigma is None else s.sigma * J, aux=aux, meta=meta,
                    x_label=x_label or cal.d.get("y_label") or s.x_label,
                    x_unit=x_unit or cal.d.get("y_unit") or "")
    if spectral_density:
        out = out.with_flags("jacobian")
    if "x_calibrated" in s.flags:
        out = warn(out, "x war bereits kalibriert – Kalibrierung wird auf die schon kalibrierte Achse angewandt.")
    cu, su = cal.d.get("x_unit") or "", s.x_unit or ""
    cl, sl = cal.d.get("x_label") or "", s.x_label or ""
    if (cu and su and cu != su) or (not (cu or su) and cl and sl and cl != sl and sl != "x"):
        out = warn(out, f"x-Achse der Daten ({sl} {su}) passt nicht zur Kalibrierung ({cl} {cu}) – Einheiten "
                        "prüfen.")
    if n_out:
        out = warn(out, f"{n_out} von {s.n} Punkten liegen außerhalb des kalibrierten Bereichs (Extrapolation).")
    return out.sorted()
