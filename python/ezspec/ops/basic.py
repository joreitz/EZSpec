"""Basic processing operations: range selection, scaling, units, resampling,
despiking, smoothing and noise estimation."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter

from .. import units
from .._backend import as_f64, core
from ..spectrum import SigmaSource, Spectrum
from .registry import ParamSpec as P
from .registry import operation, warn

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Auxiliary curves that live on the y scale and must follow y transformations.
# Everything else (independent variables "var:*", baseline weights) is left untouched.
Y_LEVEL_AUX = ("smoothed",)                          # absolute y values
Y_DIFFERENCE_AUX = ("baseline", "baseline_total")    # y differences (subtracted curves)


# =========================================================================== range
@operation("crop", 1, "Bereich beschneiden", "Bereich",
           params=[P("xmin", "float", None, "x min", optional=True),
                   P("xmax", "float", None, "x max", optional=True)])
def crop(s: Spectrum, xmin, xmax) -> Spectrum:
    """Keep only points with xmin <= x <= xmax (None = unbounded)."""
    m = np.ones(s.n, bool)
    if xmin is not None:
        m &= s.x >= xmin
    if xmax is not None:
        m &= s.x <= xmax
    if m.sum() < 2:
        raise ValueError("crop: fewer than 2 points remain")
    return s.take(m)


def _in_ranges(x, ranges):
    m = np.zeros(len(x), bool)
    for a, b in ranges or []:
        lo, hi = min(a, b), max(a, b)
        m |= (x >= lo) & (x <= hi)
    return m


@operation("exclude", 1, "Bereiche vom Fit ausschließen", "Bereich",
           params=[P("ranges", "ranges", [], "Bereiche [(a, b), ...]"),
                   P("replace", "bool", False, "vorhandene Maske ersetzen")])
def exclude(s: Spectrum, ranges, replace) -> Spectrum:
    """Mark x-ranges as excluded from fitting (data stay visible)."""
    m = _in_ranges(s.x, ranges)
    if not replace and s.exclude is not None:
        m |= s.exclude
    return s.replace(exclude=m)


# =========================================================================== scaling
@operation("offset_scale", 1, "Offset / Skalierung", "Skalierung",
           params=[P("offset", "float", 0.0, "Offset (abgezogen)"),
                   P("factor", "float", 1.0, "Faktor")])
def offset_scale(s: Spectrum, offset, factor) -> Spectrum:
    """y' = (y - offset) * factor; sigma' = sigma * |factor|."""
    if factor == 0:
        raise ValueError("factor must not be 0")
    sig = None if s.sigma is None else s.sigma * abs(factor)
    aux = dict(s.aux)
    for k in Y_LEVEL_AUX:
        if k in aux:
            aux[k] = (aux[k] - offset) * factor
    for k in Y_DIFFERENCE_AUX:
        if k in aux:
            aux[k] = aux[k] * factor
    return s.replace(y=(s.y - offset) * factor, sigma=sig, aux=aux)


@operation("normalize", 1, "Normieren", "Skalierung",
           params=[P("method", "choice", "max", "Methode",
                     choices=("max", "area", "vector", "snv", "minmax", "value_at")),
                   P("x_ref", "float", None, "x-Referenz (value_at)", optional=True)],
           flags=("normalized",))
def normalize(s: Spectrum, method, x_ref) -> Spectrum:
    """Normalisation. Absolute amplitudes/areas of later fits refer to the
    normalised scale; the normalisation constant is treated as exact."""
    y = s.y
    if method == "max":
        off, f = 0.0, 1.0 / np.nanmax(y)
    elif method == "area":
        off, f = 0.0, 1.0 / _trapezoid(y, s.x)
    elif method == "vector":
        off, f = 0.0, 1.0 / np.sqrt(np.nansum(y * y))
    elif method == "snv":
        off, f = np.nanmean(y), 1.0 / np.nanstd(y, ddof=1)
    elif method == "minmax":
        off, f = np.nanmin(y), 1.0 / (np.nanmax(y) - np.nanmin(y))
    else:  # value_at
        if x_ref is None:
            raise ValueError("value_at needs x_ref")
        off, f = 0.0, 1.0 / float(np.interp(x_ref, s.x, y))
    if not np.isfinite(f) or f == 0:
        raise ValueError("normalisation constant is zero or not finite")
    out = offset_scale.spec.func(s, offset=off, factor=f)
    meta = dict(out.meta)
    meta["normalization"] = {"method": method, "offset": float(off), "factor": float(f)}
    return out.replace(meta=meta)


# =========================================================================== units
@operation("set_units", 1, "Einheiten / Achsentitel festlegen", "Einheiten",
           params=[P("x_unit", "choice", "", "x-Einheit", choices=("",) + tuple(units.UNITS)),
                   P("x_label", "str", "", "x-Achsentitel (leer = automatisch)"),
                   P("y_unit", "str", "", "y-Einheit"),
                   P("y_label", "str", "", "y-Achsentitel")])
def set_units(s: Spectrum, x_unit, x_label, y_unit, y_label) -> Spectrum:
    """Declare units and axis labels (metadata only; data are unchanged)."""
    xl = x_label or units.AXIS_LABELS.get(x_unit, s.x_label)
    return s.replace(x_unit=x_unit, x_label=xl, y_unit=y_unit or s.y_unit, y_label=y_label or s.y_label)


@operation("convert_x", 1, "x-Einheit umrechnen", "Einheiten",
           params=[P("to", "choice", "eV", "Ziel-Einheit", choices=tuple(units.UNITS)),
                   P("from_unit", "choice", None, "Quell-Einheit (leer = aus Daten)",
                     choices=tuple(units.UNITS), optional=True),
                   P("laser_nm", "float", None, "Anregungswellenlänge / nm (Raman)", optional=True, min=0.0),
                   P("spectral_density", "bool", False, "y ist spektrale Dichte (Jacobi-Faktor)")],
           flags=())
def convert_x(s: Spectrum, to, from_unit, laser_nm, spectral_density) -> Spectrum:
    """Convert the x axis. For spectral densities y and sigma are multiplied
    by the Jacobian |dx_old/dx_new|; for ratio quantities only x changes."""
    src = from_unit or s.x_unit
    if src not in units.UNITS:
        raise ValueError(f"source unit {src!r} unknown; set from_unit")
    xn = units.convert(s.x, src, to, laser_nm)
    if spectral_density:
        J = units.jacobian(s.x, src, to, laser_nm)
    else:
        J = np.ones(s.n)
    sig = None if s.sigma is None else s.sigma * J
    aux = {k: (v * J if k in Y_LEVEL_AUX + Y_DIFFERENCE_AUX else v) for k, v in s.aux.items()}
    if "sigma_x" in aux:          # dx_new = dx_old * |dx_new/dx_old| = dx_old / |dx_old/dx_new|
        aux["sigma_x"] = aux["sigma_x"] / units.jacobian(s.x, src, to, laser_nm)
    out = s.replace(x=xn, y=s.y * J, sigma=sig, aux=aux, x_unit=to, x_label=units.AXIS_LABELS[to])
    if spectral_density:
        out = out.with_flags("jacobian")
    return out.sorted()


# =========================================================================== resampling
def _hat_operator(x, x_new, mode, edges=None):
    """Sparse matrix A with y_new = A @ y for the piecewise-linear interpolant of (x, y).

    mode="point": evaluation at x_new; mode="mean": mean over bins given by edges.
    """
    n = len(x)
    rows, cols, vals = [], [], []
    if mode == "point":
        idx = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, n - 2)
        t = (x_new - x[idx]) / (x[idx + 1] - x[idx])
        for r, (k, tk) in enumerate(zip(idx, t)):
            rows += [r, r]
            cols += [k, k + 1]
            vals += [1.0 - tk, tk]
    else:
        for r, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            k0 = max(np.searchsorted(x, lo, side="right") - 1, 0)
            k1 = min(np.searchsorted(x, hi, side="left"), n - 1)
            for k in range(k0, k1):
                a, b = max(lo, x[k]), min(hi, x[k + 1])
                if b <= a:
                    continue
                h = x[k + 1] - x[k]
                # integrals of the two hat-function pieces over [a, b]
                int_left = ((x[k + 1] - a) ** 2 - (x[k + 1] - b) ** 2) / (2 * h)
                int_right = ((b - x[k]) ** 2 - (a - x[k]) ** 2) / (2 * h)
                width = hi - lo
                rows += [r, r]
                cols += [k, k + 1]
                vals += [int_left / width, int_right / width]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(x_new), n))


@operation("resample", 1, "Neu abtasten", "Einheiten",
           params=[P("step", "float", None, "Schrittweite", optional=True, min=0.0),
                   P("n", "int", None, "Anzahl Punkte", optional=True, min=2),
                   P("method", "choice", "linear", "Methode", choices=("linear", "bin_mean"))],
           flags=("interpolated",))
def resample(s: Spectrum, step, n, method) -> Spectrum:
    """Resample onto a uniform grid. ``linear`` evaluates the linear
    interpolant; ``bin_mean`` averages it over bins (area-conserving).
    Sigma is propagated assuming independent input points; the output points
    are correlated, which the fit warns about."""
    if not s.is_strictly_increasing:
        raise ValueError("resample needs strictly increasing x")
    lo, hi = s.x[0], s.x[-1]
    if step is not None:
        m = int(np.floor((hi - lo) / step + 1e-9)) + 1
    elif n is not None:
        m = n
    else:
        raise ValueError("give either step or n")
    if method == "linear":
        xn = np.linspace(lo, lo + (m - 1) * (step if step else (hi - lo) / (m - 1)), m)
        A = _hat_operator(s.x, xn, "point")
    else:
        edges = np.linspace(lo, hi, m + 1)
        xn = 0.5 * (edges[:-1] + edges[1:])
        A = _hat_operator(s.x, xn, "mean", edges)
    yn = A @ s.y
    sig = None if s.sigma is None else np.sqrt(A.multiply(A) @ (s.sigma**2))
    aux = {k: A @ v for k, v in s.aux.items() if k != "acq_index"}     # acquisition order is lost
    return Spectrum(xn, yn, sig, s.sigma_source if sig is not None else SigmaSource.UNKNOWN,
                    s.x_unit, s.y_unit, s.x_label, s.y_label, None, s.meta, s.flags, aux)


# =========================================================================== despiking
def whitaker_hayes_spikes(y, threshold):
    """Spike mask after Whitaker & Hayes (2018): modified z-score of the first differences."""
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return np.zeros(len(y), bool)
    z = 0.6745 * (d - med) / mad
    spikes = np.zeros(len(y), bool)
    spikes[1:] = np.abs(z) > threshold
    return spikes


@operation("despike", 1, "Spikes entfernen (Whitaker–Hayes)", "Korrektur",
           params=[P("threshold", "float", 8.0, "Schwelle (modif. z-Score)", min=0.0),
                   P("kernel", "int", 3, "Fenster ± Punkte", min=1)],
           flags=("despiked",))
def despike(s: Spectrum, threshold, kernel) -> Spectrum:
    """Cosmic-ray removal: points whose first difference has a modified
    z-score above ``threshold`` are replaced by the mean of the non-spike
    points within +/- ``kernel``. Defaults follow RamanSPy (kernel 3, threshold 8)."""
    spikes = whitaker_hayes_spikes(s.y, threshold)
    y = s.y.copy()
    for i in np.flatnonzero(spikes):
        lo, hi = max(0, i - kernel), min(s.n, i + kernel + 1)
        good = ~spikes[lo:hi]
        if good.any():
            y[i] = s.y[lo:hi][good].mean()
    out = s.replace(y=y)
    meta = dict(out.meta)
    meta["despike_replaced"] = int(spikes.sum())
    out = out.replace(meta=meta)
    if spikes.sum() > 0.05 * s.n:
        out = warn(out, f"Despike ersetzte {spikes.sum()} Punkte (>5 %) – Schwelle zu niedrig?")
    return out


# =========================================================================== smoothing
_SMOOTH_NOTE = ("Glätten senkt Peakhöhen, verbreitert Peaks und korreliert das Rauschen; "
                "Fits auf geglätteten Daten unterschätzen Unsicherheiten stark (O'Haver).")


def _smoothed(s: Spectrum, ys: np.ndarray, target: str) -> Spectrum:
    if target == "display":
        aux = dict(s.aux)
        aux["smoothed"] = ys
        return s.replace(aux=aux)
    return s.replace(y=ys).with_flags("smoothed")


_TARGET = P("target", "choice", "display", "anwenden auf", choices=("display", "data"),
            help="display: nur Anzeige/Peaksuche (empfohlen); data: ersetzt y")


@operation("smooth_moving_average", 1, "Gleitender Mittelwert", "Glätten",
           params=[P("half_window", "int", 2, "Halbfenster (Punkte)", min=1),
                   P("edges", "choice", "shrink", "Ränder", choices=("shrink", "reflect", "nearest")),
                   _TARGET],
           description="Zentrierter gleitender Mittelwert. " + _SMOOTH_NOTE)
def smooth_moving_average(s: Spectrum, half_window, edges, target) -> Spectrum:
    return _smoothed(s, core().moving_average(as_f64(s.y), half_window, edges), target)


@operation("smooth_savgol", 1, "Savitzky–Golay", "Glätten",
           params=[P("window", "int", 11, "Fensterlänge (ungerade)", min=3),
                   P("polyorder", "int", 3, "Polynomgrad", min=0),
                   _TARGET],
           description="Savitzky–Golay-Filter (scipy, Randmodus 'interp'). " + _SMOOTH_NOTE)
def smooth_savgol(s: Spectrum, window, polyorder, target) -> Spectrum:
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        raise ValueError("polyorder must be < window")
    return _smoothed(s, savgol_filter(s.y, window, polyorder, mode="interp"), target)


@operation("smooth_whittaker", 1, "Whittaker-Glättung", "Glätten",
           params=[P("lam", "log_float", 1e2, "λ", min=0.0),
                   P("diff_order", "int", 2, "Differenzenordnung", min=1, max=3),
                   _TARGET],
           description="Whittaker-Glätter (gute Nebenkeulendämpfung). " + _SMOOTH_NOTE)
def smooth_whittaker(s: Spectrum, lam, diff_order, target) -> Spectrum:
    ys = core().whittaker_smooth(as_f64(s.y), np.ones(s.n), lam, diff_order)
    return _smoothed(s, ys, target)


# =========================================================================== noise / sigma
def detrended_std(x, y):
    """Standard deviation of y around a straight-line fit (ddof = 2)."""
    if len(x) < 4:
        raise ValueError("need at least 4 points in the noise region")
    coef = np.polyfit(x - x.mean(), y, 1)
    r = y - np.polyval(coef, x - x.mean())
    return float(np.sqrt(np.sum(r * r) / (len(x) - 2)))


@operation("estimate_noise", 1, "Rauschen schätzen → σ", "Unsicherheit",
           params=[P("method", "choice", "der_snr", "Methode", choices=("der_snr", "region")),
                   P("region", "range", [None, None], "flacher Bereich (region)")])
def estimate_noise(s: Spectrum, method, region) -> Spectrum:
    """Set a constant sigma estimated from the data. ``der_snr``:
    0.6052697 * median|2y_i - y_{i-2} - y_{i+2}| (assumes white noise and a
    signal locally linear over 5 points). ``region``: standard deviation
    around a straight line in a user-marked signal-free region."""
    if method == "der_snr":
        sig = core().der_snr(as_f64(s.y))
        src = SigmaSource.ESTIMATED_DERSNR
    else:
        lo, hi = region
        m = np.ones(s.n, bool)
        if lo is not None:
            m &= s.x >= min(lo, hi if hi is not None else lo)
        if hi is not None:
            m &= s.x <= max(hi, lo if lo is not None else hi)
        sig = detrended_std(s.x[m], s.y[m])
        src = SigmaSource.ESTIMATED_REGION
    if not (np.isfinite(sig) and sig > 0):
        raise ValueError("noise estimate is zero or undefined")
    meta = dict(s.meta)
    meta["noise_estimate"] = {"method": method, "sigma": float(sig)}
    return s.replace(sigma=np.full(s.n, sig), sigma_source=src, meta=meta)


@operation("set_sigma", 1, "σ festlegen", "Unsicherheit",
           params=[P("mode", "choice", "constant", "Modus", choices=("constant", "poisson_data", "none")),
                   P("value", "float", 1.0, "Wert (constant)", min=0.0)])
def set_sigma(s: Spectrum, mode, value) -> Spectrum:
    """Set sigma explicitly. ``poisson_data`` (sigma = sqrt(max(y, 1))) is the
    Neyman approximation; it biases amplitudes and areas downwards even at
    high counts (Humphrey et al. 2009) – prefer the Poisson weighting of the
    fit (sigma^2 = model), which converges to the Poisson maximum likelihood."""
    if mode == "none":
        return s.replace(sigma=None, sigma_source=SigmaSource.UNKNOWN)
    if mode == "constant":
        if not value > 0:
            raise ValueError("value must be > 0")
        return s.replace(sigma=np.full(s.n, value), sigma_source=SigmaSource.CONSTANT)
    out = s.replace(sigma=np.sqrt(np.maximum(s.y, 1.0)), sigma_source=SigmaSource.POISSON_DATA)
    return warn(out, "σ = √y ist verzerrt (Humphrey et al. 2009); besser Fit-Gewichtung 'Poisson (σ² = Modell)'.")
