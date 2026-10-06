"""Library of classic fit functions (as editable formulas) and peak helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.signal import peak_widths

from .. import lineshapes as ls
from .model import Model


@dataclass(frozen=True)
class Template:
    name: str
    expression: str
    defaults: dict
    description: str = ""
    category: str = "Allgemein"
    bounds: dict | None = None


TEMPLATES = [
    Template("Gerade", "a + b*x", {"a": 0.0, "b": 1.0}, "y = a + b·x", "Polynome"),
    Template("Parabel", "a + b*x + c2*x^2", {"a": 0.0, "b": 1.0, "c2": 0.0}, "Polynom 2. Grades", "Polynome"),
    Template("Kubisch", "a + b*x + c2*x^2 + c3*x^3", {"a": 0.0, "b": 1.0, "c2": 0.0, "c3": 0.0},
             "Polynom 3. Grades", "Polynome"),
    Template("Exp. Zerfall (1)", "y0 + A*exp(-(x - x0)/tau)", {"y0": 0.0, "A": 1.0, "x0": 0.0, "tau": 1.0},
             "x0 ist fest zu halten (sonst mit A vollständig korreliert)", "Exponentiell",
             {"tau": (0.0, None)}),
    Template("Exp. Zerfall (2)", "y0 + A1*exp(-x/tau1) + A2*exp(-x/tau2)",
             {"y0": 0.0, "A1": 1.0, "tau1": 1.0, "A2": 0.5, "tau2": 10.0},
             "Biexponentiell; lineare Standardfehler oft unzuverlässig → Profil-CI nutzen", "Exponentiell",
             {"tau1": (0.0, None), "tau2": (0.0, None)}),
    Template("Exp. Wachstum", "y0 + A*exp(x/t)", {"y0": 0.0, "A": 1.0, "t": 1.0}, "", "Exponentiell"),
    Template("Gestreckte Exponentialfkt.", "y0 + A*exp(-(x/tau)^beta)",
             {"y0": 0.0, "A": 1.0, "tau": 1.0, "beta": 1.0}, "Kohlrausch–Williams–Watts", "Exponentiell",
             {"tau": (0.0, None), "beta": (0.0, 1.0)}),
    Template("Potenzgesetz", "A*x^b", {"A": 1.0, "b": 1.0}, "y = A·x^b (x > 0)", "Allgemein"),
    Template("Logarithmus", "a + b*ln(x)", {"a": 0.0, "b": 1.0}, "", "Allgemein"),
    Template("Boltzmann-Sigmoid", "A2 + (A1 - A2)/(1 + exp((x - x0)/dx))",
             {"A1": 0.0, "A2": 1.0, "x0": 0.0, "dx": 1.0}, "", "Sigmoidal"),
    Template("Logistisch (4PL)", "A2 + (A1 - A2)/(1 + (x/x0)^p)", {"A1": 0.0, "A2": 1.0, "x0": 1.0, "p": 1.0},
             "Dosis-Wirkung", "Sigmoidal"),
    Template("Hill", "y0 + (ymax - y0)*x^n/(k^n + x^n)", {"y0": 0.0, "ymax": 1.0, "k": 1.0, "n": 1.0},
             "", "Sigmoidal"),
    Template("Michaelis–Menten", "vmax*x/(km + x)", {"vmax": 1.0, "km": 1.0}, "", "Kinetik"),
    Template("Arrhenius", "A*exp(-Ea/(8.314462618*x))", {"A": 1.0, "Ea": 1e4},
             "x = T in K, Ea in J/mol", "Kinetik"),
    Template("Sinus", "y0 + A*sin(2*pi*f*x + phi)", {"y0": 0.0, "A": 1.0, "f": 1.0, "phi": 0.0}, "",
             "Periodisch"),
    Template("Gedämpfte Schwingung", "y0 + A*exp(-x/tau)*sin(2*pi*f*x + phi)",
             {"y0": 0.0, "A": 1.0, "tau": 1.0, "f": 1.0, "phi": 0.0}, "", "Periodisch"),
    Template("Gauß (Höhe, σ)", "y0 + h*exp(-(x - xc)^2/(2*s^2))", {"y0": 0.0, "h": 1.0, "xc": 0.0, "s": 1.0},
             "Origin-artige Parametrisierung (Höhe statt Fläche)", "Peaks"),
    Template("Lorentz (Höhe)", "y0 + h/(1 + ((x - xc)/g)^2)", {"y0": 0.0, "h": 1.0, "xc": 0.0, "g": 1.0},
             "g = halbe Halbwertsbreite", "Peaks"),
]
TEMPLATE_BY_NAME = {t.name: t for t in TEMPLATES}


def add_template(model: Model, name: str, prefix: str = "") -> None:
    t = TEMPLATE_BY_NAME[name]
    c = model.add("formula", prefix=prefix,
                  options={"expression": t.expression, "defaults": dict(t.defaults)}, label=t.name)
    for p, (lo, hi) in (t.bounds or {}).items():
        if lo is not None:
            c.settings[p].min = lo
        if hi is not None:
            c.settings[p].max = hi
    if t.name.startswith("Exp. Zerfall (1)"):
        c.settings["x0"].vary = False


# ----------------------------------------------------------------------------- peaks
HEIGHT_TO_AREA = {  # area = height * fwhm * factor
    "gaussian": 1.0 / (2 * np.sqrt(np.log(2) / np.pi)),   # sqrt(pi / (4 ln 2)) = 1.0645
    "lorentzian": np.pi / 2,
}


def area_from_height(kind: str, height: float, fwhm: float, eta: float = 0.5) -> float:
    g = HEIGHT_TO_AREA["gaussian"]
    l = HEIGHT_TO_AREA["lorentzian"]
    if kind == "gaussian":
        return height * fwhm * g
    if kind == "lorentzian":
        return height * fwhm * l
    if kind in ("voigt", "tch_pseudo_voigt"):
        # FWHM split equally into Gaussian and Lorentzian parts
        return height / ls.voigt(0.0, 1.0, 0.0, fwhm / 1.6376, fwhm / 1.6376) if kind == "voigt" \
            else height / ls.tch_pseudo_voigt(0.0, 1.0, 0.0, fwhm / 1.6376, fwhm / 1.6376)
    if kind == "pseudo_voigt":
        return height * fwhm / (eta / l + (1 - eta) / g)
    if kind == "pearson7":
        return height / ls.pearson7(0.0, 1.0, 0.0, fwhm, 2.0)
    if kind == "emg":
        return height * fwhm * g
    raise ValueError(kind)


def add_peak(model: Model, kind: str, center: float, height: float, fwhm: float,
             prefix: str | None = None, bounds_window: float | None = None):
    """Add a peak from intuitive start values (centre, height, FWHM).

    ``bounds_window`` (in x units) optionally restricts the centre to
    centre +/- window, which stabilises multi-peak fits.
    """
    area = area_from_height(kind, height, fwhm)
    values = {"area": area}
    if kind == "emg":
        values.update(mu=center, fwhm_g=fwhm, tau=0.2 * fwhm)
    else:
        values["center"] = center
    if kind in ("voigt", "tch_pseudo_voigt"):
        values.update(fwhm_g=fwhm / 1.6376, fwhm_l=fwhm / 1.6376)
    elif kind != "emg":
        values["fwhm"] = fwhm
    comp = model.add(kind, prefix=prefix, **values)
    if bounds_window:
        key = "mu" if kind == "emg" else "center"
        comp.settings[key].min = center - bounds_window
        comp.settings[key].max = center + bounds_window
    return comp


@dataclass
class PeakGuess:
    center: float
    height: float
    fwhm: float
    prominence: float


def find_peaks(x, y, prominence: float | None = None, min_fwhm: float | None = None,
               max_peaks: int = 20, smooth_half_window: int = 2) -> list:
    """Start values from local maxima with a prominence criterion (scipy.signal.find_peaks).

    The search runs on a lightly smoothed copy (moving average, only for
    detection). Default prominence: 5 x the DER_SNR noise estimate of the raw
    data. The widths are the widths at half prominence
    (scipy.signal.peak_widths, rel_height = 0.5). These are start values;
    the fit uses the unsmoothed data.
    """
    from .._backend import as_f64, core
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ys = core().moving_average(as_f64(y), smooth_half_window, "shrink") if smooth_half_window else y
    if prominence is None:
        noise = core().der_snr(as_f64(y))
        prominence = 5 * noise if np.isfinite(noise) and noise > 0 else 0.05 * np.ptp(y)
    idx, props = _find_peaks(ys, prominence=prominence)
    if len(idx) == 0:
        return []
    widths = peak_widths(ys, idx, rel_height=0.5)[0]
    dx = np.gradient(x)
    out = []
    for i, w, prom in zip(idx, widths, props["prominences"]):
        fwhm = float(w * dx[i])
        if min_fwhm is not None and fwhm < min_fwhm:
            continue
        out.append(PeakGuess(float(x[i]), float(prom), fwhm, float(prom)))
    out.sort(key=lambda p: -p.prominence)
    return sorted(out[:max_peaks], key=lambda p: p.center)
