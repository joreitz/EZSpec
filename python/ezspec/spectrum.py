"""The central data type.

A :class:`Spectrum` is immutable. Processing steps return new spectra; the
original data are never overwritten. Besides x and y it carries

* ``sigma`` – per-point standard uncertainty of y (or ``None``),
* ``sigma_source`` – *where* sigma comes from. This is mandatory metadata:
  a chi-square is only a chi-square if sigma is known, so every fit result
  reports it,
* units, an exclusion mask for fitting, free-form metadata,
* ``flags`` – processing history relevant to statistics (e.g. ``smoothed``),
  used to warn when a fit is run on data whose noise is no longer
  independent,
* ``aux`` – auxiliary curves produced by processing (e.g. the subtracted
  baseline), kept for plotting and export.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np


class SigmaSource(str, Enum):
    """Provenance of the y uncertainties."""

    KNOWN = "known"                    # supplied with the data (column)
    ESTIMATED_DERSNR = "der_snr"       # estimated from the data (DER_SNR)
    ESTIMATED_REGION = "region"        # std. dev. of a user-marked flat region
    CONSTANT = "constant"              # constant entered by the user
    POISSON_DATA = "poisson_data"      # sigma = sqrt(y)  (biased, discouraged)
    UNKNOWN = "unknown"                # no sigma; fit uses unit weights

    @property
    def label(self) -> str:
        return SIGMA_LABELS[self]

    @property
    def is_known(self) -> bool:
        """True if sigma has an absolute scale (chi-square interpretable)."""
        return self in (SigmaSource.KNOWN, SigmaSource.ESTIMATED_DERSNR,
                        SigmaSource.ESTIMATED_REGION, SigmaSource.CONSTANT,
                        SigmaSource.POISSON_DATA)


SIGMA_LABELS = {
    SigmaSource.KNOWN: "bekannt (Datenspalte)",
    SigmaSource.ESTIMATED_DERSNR: "geschätzt (DER_SNR)",
    SigmaSource.ESTIMATED_REGION: "geschätzt (flacher Bereich)",
    SigmaSource.CONSTANT: "konstant (vom Nutzer gesetzt)",
    SigmaSource.POISSON_DATA: "Poisson σ = √y (verzerrt!)",
    SigmaSource.UNKNOWN: "unbekannt",
}


def _frozen(a, dtype=float):
    if a is None:
        return None
    arr = np.array(a, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class Spectrum:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | None = None
    sigma_source: SigmaSource = SigmaSource.UNKNOWN
    x_unit: str = ""
    y_unit: str = ""
    x_label: str = "x"
    y_label: str = "y"
    exclude: np.ndarray | None = None          # True = point excluded from fits
    meta: Mapping[str, Any] = field(default_factory=dict)
    flags: frozenset = frozenset()
    aux: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        x = _frozen(self.x)
        y = _frozen(self.y)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError(f"x and y must be 1-D arrays of equal length, got {x.shape}, {y.shape}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        sigma = _frozen(self.sigma)
        if sigma is not None:
            if sigma.shape != x.shape:
                raise ValueError("sigma must have the same length as x")
            if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
                raise ValueError("sigma must be finite and > 0 everywhere")
        object.__setattr__(self, "sigma", sigma)
        src = SigmaSource(self.sigma_source)
        if sigma is None:
            src = SigmaSource.UNKNOWN
        elif src is SigmaSource.UNKNOWN:
            raise ValueError("a sigma array requires a sigma_source other than UNKNOWN")
        object.__setattr__(self, "sigma_source", src)
        excl = _frozen(self.exclude, bool)
        if excl is not None and excl.shape != x.shape:
            raise ValueError("exclude mask must have the same length as x")
        object.__setattr__(self, "exclude", excl)
        object.__setattr__(self, "meta", dict(self.meta))
        object.__setattr__(self, "flags", frozenset(self.flags))
        object.__setattr__(self, "aux", {k: _frozen(v) for k, v in dict(self.aux).items()})

    # ------------------------------------------------------------------ helpers
    @property
    def n(self) -> int:
        return len(self.x)

    def __len__(self) -> int:
        return self.n

    def replace(self, **changes) -> "Spectrum":
        return replace(self, **changes)

    def with_flags(self, *flags: str) -> "Spectrum":
        return self.replace(flags=self.flags | set(flags))

    @property
    def fit_mask(self) -> np.ndarray:
        """Boolean mask of points used for fitting (finite and not excluded)."""
        m = np.isfinite(self.x) & np.isfinite(self.y)
        if self.exclude is not None:
            m &= ~self.exclude
        return m

    def take(self, idx) -> "Spectrum":
        """Subset by index array or boolean mask (aux curves are subset too)."""
        idx = np.asarray(idx)
        return self.replace(
            x=self.x[idx], y=self.y[idx],
            sigma=None if self.sigma is None else self.sigma[idx],
            exclude=None if self.exclude is None else self.exclude[idx],
            aux={k: v[idx] for k, v in self.aux.items() if len(v) == self.n},
        )

    def sorted(self) -> "Spectrum":
        if np.all(np.diff(self.x) >= 0):
            return self
        return self.take(np.argsort(self.x, kind="stable"))

    @property
    def is_strictly_increasing(self) -> bool:
        return bool(np.all(np.diff(self.x) > 0))

    @property
    def dx_median(self) -> float:
        return float(np.median(np.diff(self.x))) if self.n > 1 else float("nan")

    def content_hash(self) -> str:
        """SHA-256 over x, y, sigma, exclusion mask and sigma source."""
        h = hashlib.sha256()
        for a in (self.x, self.y, self.sigma, self.exclude):
            if a is None:
                h.update(b"\x00none")
            else:
                h.update(str(a.dtype).encode())
                h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.sigma_source.value.encode())
        return h.hexdigest()

    def __repr__(self) -> str:
        unit = f" [{self.x_unit}]" if self.x_unit else ""
        return (f"Spectrum(n={self.n}, x={self.x[0]:.6g}..{self.x[-1]:.6g}{unit}, "
                f"sigma={self.sigma_source.value}, flags={sorted(self.flags)})") if self.n else "Spectrum(empty)"


def spectrum(x, y, sigma=None, **kw) -> Spectrum:
    """Convenience constructor; data are sorted by x. A sigma array without
    explicit ``sigma_source`` is treated as known (supplied with the data)."""
    if sigma is not None and "sigma_source" not in kw:
        kw["sigma_source"] = SigmaSource.KNOWN
    return Spectrum(np.asarray(x, float), np.asarray(y, float), sigma, **kw).sorted()
