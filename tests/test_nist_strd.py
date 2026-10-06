"""Validation against the NIST Statistical Reference Datasets for nonlinear regression.

All 27 problems are fitted through the full EZSpec path (formula parser ->
Model -> fit engine, unit weights, scaled covariance, i.e. sigma unknown as
in the NIST certification). From the better of the two NIST start vectors we
require at least 7 significant digits for every certified parameter value,
9 digits for the residual sum of squares and 6 digits for the certified
standard deviations. Exception: Lanczos1 has a certified RSS of 1.4e-25,
i.e. its RSS and standard deviations are pure floating-point round-off;
there only the parameter values are checked. The data files are public-domain NIST data (copied from the
lmfit source distribution).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from ezspec import Model, spectrum
from ezspec.fit import FitOptions, fit

DATA = Path(__file__).parent / "data" / "nist"

MODELS = {
    "Bennett5": "b1 * (b2 + x)**(-1/b3)",
    "BoxBOD": "b1*(1 - exp(-b2*x))",
    "Chwirut1": "exp(-b1*x)/(b2 + b3*x)",
    "Chwirut2": "exp(-b1*x)/(b2 + b3*x)",
    "DanWood": "b1*x**b2",
    "ENSO": ("b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12) + b5*cos(2*pi*x/b4) + b6*sin(2*pi*x/b4)"
             " + b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)"),
    "Eckerle4": "(b1/b2) * exp(-0.5*((x - b3)/b2)**2)",
    "Gauss1": "b1*exp(-b2*x) + b3*exp(-(x - b4)**2/b5**2) + b6*exp(-(x - b7)**2/b8**2)",
    "Gauss2": "b1*exp(-b2*x) + b3*exp(-(x - b4)**2/b5**2) + b6*exp(-(x - b7)**2/b8**2)",
    "Gauss3": "b1*exp(-b2*x) + b3*exp(-(x - b4)**2/b5**2) + b6*exp(-(x - b7)**2/b8**2)",
    "Hahn1": "(b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)",
    "Kirby2": "(b1 + b2*x + b3*x**2)/(1 + b4*x + b5*x**2)",
    "Lanczos1": "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Lanczos2": "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Lanczos3": "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "MGH09": "b1*(x**2 + x*b2)/(x**2 + x*b3 + b4)",
    "MGH10": "b1*exp(b2/(x + b3))",
    "MGH17": "b1 + b2*exp(-x*b4) + b3*exp(-x*b5)",
    "Misra1a": "b1*(1 - exp(-b2*x))",
    "Misra1b": "b1*(1 - (1 + b2*x/2)**(-2))",
    "Misra1c": "b1*(1 - (1 + 2*b2*x)**(-0.5))",
    "Misra1d": "b1*b2*x*((1 + b2*x)**(-1))",
    "Nelson": "b1 - b2*x1*exp(-b3*x2)",
    "Rat42": "b1/(1 + exp(b2 - b3*x))",
    "Rat43": "b1/((1 + exp(b2 - b3*x))**(1/b4))",
    "Roszman1": "b1 - b2*x - arctan(b3/(x - b4))/pi",
    "Thurber": "(b1 + b2*x + b3*x**2 + b4*x**3)/(1 + b5*x + b6*x**2 + b7*x**3)",
}


def read_nist(name):
    lines = (DATA / f"{name}.dat").read_text().splitlines()
    start1, start2, cert, cert_sd = [], [], [], []
    rss = None
    data_start = None
    for i, line in enumerate(lines):
        t = line.split()
        if len(t) >= 6 and t[0].startswith("b") and t[1] == "=":
            start1.append(float(t[2]))
            start2.append(float(t[3]))
            cert.append(float(t[4]))
            cert_sd.append(float(t[5]))
        if line.startswith("Residual Sum of Squares"):
            rss = float(t[-1])
        if line.startswith("Data:") and ("y" in t[1:2] or t[1:2] == ["y"]):
            data_start = i + 1
    rows = np.array([[float(v) for v in ln.split()] for ln in lines[data_start:] if ln.strip()])
    return {"start1": start1, "start2": start2, "cert": cert, "cert_sd": cert_sd, "rss": rss, "rows": rows}


def digits(a, b):
    if b == 0:
        return 15 if abs(a) < 1e-15 else 0
    rel = abs(a - b) / abs(b)
    return 15 if rel == 0 else min(15, -math.log10(rel))


def run(name, start):
    d = read_nist(name)
    rows = d["rows"]
    if name == "Nelson":
        s = spectrum(rows[:, 1], np.log(rows[:, 0]), aux={"var:x2": rows[:, 2]})
        independent = ["x1", "x2"]
    else:
        s = spectrum(rows[:, 1], rows[:, 0])
        independent = ["x"]
    starts = d[start]
    m = Model()
    m.add("formula", options={"expression": MODELS[name], "independent": independent,
                              "defaults": {f"b{i + 1}": v for i, v in enumerate(starts)}})
    r = fit(s, m, FitOptions(method="leastsq", weighting="none"))
    vd = min(digits(r.value(f"b{i + 1}"), c) for i, c in enumerate(d["cert"]))
    if r.stats.covariance_mode == "unavailable":
        ed = 0
    else:
        ed = min(digits(r.stderr(f"b{i + 1}"), c) for i, c in enumerate(d["cert_sd"]))
    rd = digits(r.stats.rss, d["rss"])
    return vd, ed, rd, r


def test_reader_and_models_reproduce_certified_rss():
    for name in MODELS:
        d = read_nist(name)
        rows = d["rows"]
        m = Model()
        indep = ["x1", "x2"] if name == "Nelson" else ["x"]
        m.add("formula", options={"expression": MODELS[name], "independent": indep})
        vals = {f"b{i + 1}": v for i, v in enumerate(d["cert"])}
        if name == "Nelson":
            f = m.evaluate(rows[:, 1], vals, {"x2": rows[:, 2]})
            y = np.log(rows[:, 0])
        else:
            f = m.evaluate(rows[:, 1], vals)
            y = rows[:, 0]
        rss = float(np.sum((y - f) ** 2))
        assert rss == pytest.approx(d["rss"], rel=1e-8, abs=1e-20), name


@pytest.mark.parametrize("name", sorted(MODELS))
def test_nist_certified_values(name):
    best = None
    for start in ("start1", "start2"):
        try:
            vd, ed, rd, r = run(name, start)
        except Exception:  # noqa: BLE001 - a failing start is allowed if the other succeeds
            continue
        score = vd
        if best is None or score > best[0]:
            best = (score, vd, ed, rd, start)
    assert best is not None, f"{name}: both starts failed"
    _, vd, ed, rd, start = best
    assert vd >= 7, f"{name} ({start}): only {vd:.1f} digits in parameter values"
    if name == "Lanczos1":
        return
    assert rd >= 9, f"{name} ({start}): only {rd:.1f} digits in RSS"
    assert ed >= 6, f"{name} ({start}): only {ed:.1f} digits in standard errors"
