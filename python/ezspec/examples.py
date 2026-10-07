"""Synthetic example data with known ground truth (for demos and tests)."""

from __future__ import annotations

import numpy as np

from . import lineshapes as ls
from .spectrum import Spectrum

RAMAN_TRUTH = {
    "peaks": [("voigt", 1001.0, 900.0, 6.0, 3.0), ("lorentzian", 1032.0, 380.0, 9.0, None),
              ("gaussian", 1157.0, 300.0, 14.0, None), ("lorentzian", 1602.0, 700.0, 12.0, None)],
    "noise": 1.5,
}


def raman_example(seed: int = 7, n: int = 1500, spike: bool = True) -> Spectrum:
    """Raman-like spectrum on a broad fluorescence background with Gaussian noise
    (sigma = 1.5) and one cosmic-ray spike. Ground truth: ``RAMAN_TRUTH``."""
    rng = np.random.default_rng(seed)
    x = np.linspace(400.0, 1800.0, n)
    background = 80.0 + 60.0 * np.exp(-((x - 1300.0) / 700.0) ** 2) + 0.02 * (x - 400.0)
    y = background.copy()
    for kind, c, area, w1, w2 in RAMAN_TRUTH["peaks"]:
        if kind == "voigt":
            y += ls.voigt(x, area, c, w1, w2)
        elif kind == "lorentzian":
            y += ls.lorentzian(x, area, c, w1)
        else:
            y += ls.gaussian(x, area, c, w1)
    y += rng.normal(scale=RAMAN_TRUTH["noise"], size=n)
    if spike:
        y[int(0.63 * n)] += 250.0
    return Spectrum(x, y, x_unit="raman", y_unit="counts", x_label="Raman shift (cm⁻¹)", y_label="Intensity",
                    meta={"name": "Example: Raman (synthetic)", "truth": "ezspec.examples.RAMAN_TRUTH"})


def decay_example(seed: int = 3, n: int = 200) -> Spectrum:
    """Biexponential decay with Poisson counts (tau = 2.0 and 15.0)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 60.0, n)
    lam = 5.0 + 800.0 * np.exp(-t / 2.0) + 200.0 * np.exp(-t / 15.0)
    return Spectrum(t, rng.poisson(lam).astype(float), x_unit="", y_unit="counts", x_label="t (ns)",
                    y_label="Counts", meta={"name": "Example: decay curve (Poisson)"})


def to_csv_bytes(s: Spectrum) -> bytes:
    """Exact (repr) CSV serialisation, readable by :func:`ezspec.io.read_file`."""
    head = f"{s.x_label},{s.y_label}"
    lines = [head] + [f"{a!r},{b!r}" for a, b in zip(s.x.tolist(), s.y.tolist())]
    return ("\n".join(lines) + "\n").encode("utf-8")
