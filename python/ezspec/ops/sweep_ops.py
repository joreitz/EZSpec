"""Operations for back-and-forth (ramp) measurements."""

from __future__ import annotations

import numpy as np

from ..spectrum import SigmaSource, Spectrum
from ..sweeps import find_sweeps
from .registry import ParamSpec as P
from .registry import operation, warn

_DIR = {"up": 1, "down": -1}
_COMMON = [P("tolerance", "float", 0.01, "Reversal threshold (fraction of x range)", min=0.0, max=0.5,
             help="A change of direction is only registered once x runs back by more than this fraction (robust against noise)"),
           P("min_points", "int", 5, "Min. points per sweep", min=2)]


def _describe(sw):
    up = sum(1 for w in sw if w.direction == 1)
    down = sum(1 for w in sw if w.direction == -1)
    flat = sum(1 for w in sw if w.direction == 0)
    return {"up": up, "down": down, "flat": flat}


def _merge_duplicates(x, y, e):
    """Average points of one ramp that share an x value (e.g. a held turning
    point) so the ramp can be interpolated; sigma of the mean is propagated."""
    ux, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if len(ux) == len(x):
        return x, y, e
    ym = np.bincount(inv, weights=y) / cnt
    em = None if e is None else np.sqrt(np.bincount(inv, weights=e ** 2)) / cnt
    return ux, ym, em


@operation("select_sweeps", 1, "Select sweeps (up/down)", "Range",
           params=[P("direction", "choice", "up", "Direction", choices=("up", "down", "all")),
                   P("sweep", "int", -1, "No. (0, 1, …; −1 = all in this direction)", min=-1)] + _COMMON)
def select_sweeps(s: Spectrum, direction, sweep, tolerance, min_points) -> Spectrum:
    """Keep only the ramps of one direction (or a single ramp). Up and down
    ramps are not mixed: their x scales can differ (hysteresis)."""
    sw = find_sweeps(s, tolerance, min_points)
    info = _describe(sw)
    if direction == "all":
        chosen = [w for w in sw if w.direction != 0]
        if sweep >= 0:
            chosen = chosen[sweep:sweep + 1]
    else:
        chosen = [w for w in sw if w.direction == _DIR[direction]]
        if sweep >= 0:
            chosen = [w for w in chosen if w.number == sweep]
    if not chosen:
        raise ValueError(f"no matching sweep found (found: {info['up']} up, "
                         f"{info['down']} down, {info['flat']} flat)")
    idx = np.sort(np.concatenate([w.indices for w in chosen]))
    out = s.take(idx)
    meta = dict(out.meta)
    meta["sweeps"] = {**info, "selected": direction, "sweep": sweep, "n_selected": len(chosen)}
    out = out.replace(meta=meta)
    dirs = {w.direction for w in chosen}
    if len(chosen) > 1:
        out = warn(out, f"{len(chosen)} sweeps selected – x remains ambiguous; select a single sweep or use "
                        "'Average sweeps'.")
    if dirs == {1, -1}:
        out = warn(out, "Up and down ramps mixed – with hysteresis their x scales differ.")
    return out


@operation("average_sweeps", 1, "Average sweeps (same direction)", "Range",
           params=[P("direction", "choice", "up", "Direction", choices=("up", "down")),
                   P("sigma", "choice", "repeats", "σ of the mean",
                     choices=("repeats", "propagate", "none"),
                     help="repeats: scatter of the sweeps / √n; propagate: from the existing σ")] + _COMMON)
def average_sweeps(s: Spectrum, direction, sigma, tolerance, min_points) -> Spectrum:
    """Average all ramps of one direction point by point. Identical x grids are
    averaged directly, otherwise the ramps are interpolated onto the grid of the
    first ramp (flagged as interpolated). With sigma = repeats the uncertainty
    is the standard error of the mean of the repetitions."""
    sw = [w for w in find_sweeps(s, tolerance, min_points) if w.direction == _DIR[direction]]
    if len(sw) < 2:
        raise ValueError(f"at least two {'up' if direction == 'up' else 'down'} sweeps required, "
                         f"found: {len(sw)}")
    segs = []
    for w in sw:
        o = np.argsort(s.x[w.indices], kind="stable")
        ii = w.indices[o]
        segs.append((s.x[ii], s.y[ii], None if s.sigma is None else s.sigma[ii]))
    x0 = segs[0][0]
    same = all(len(x) == len(x0) and np.allclose(x, x0, rtol=0, atol=1e-9 * max(np.ptp(x0), 1e-300))
               for x, _, _ in segs)
    interpolated = False
    if same:
        X = x0
        Y = np.vstack([y for _, y, _ in segs])
        S = None if s.sigma is None else np.vstack([e for _, _, e in segs])
    else:
        lo = max(x.min() for x, _, _ in segs)
        hi = min(x.max() for x, _, _ in segs)
        X = x0[(x0 >= lo) & (x0 <= hi)]
        if len(X) < 2:
            raise ValueError("the sweeps do not overlap in x")
        segs = [_merge_duplicates(*seg) for seg in segs]
        Y = np.vstack([np.interp(X, x, y) for x, y, _ in segs])
        S = None if s.sigma is None else np.vstack([np.interp(X, x, e) for x, _, e in segs])
        interpolated = True
    n = Y.shape[0]
    mean = Y.mean(axis=0)
    out_sigma, src = None, SigmaSource.UNKNOWN
    msgs = []
    if sigma == "repeats":
        sd = Y.std(axis=0, ddof=1)
        if np.all(sd > 0):
            out_sigma, src = sd / np.sqrt(n), SigmaSource.REPEATS
        else:
            msgs.append("Scatter of the sweeps is 0 at some points – no σ")
        if n < 5:
            msgs.append(f"σ estimated from only {n} repeats: relative uncertainty of σ ≈ "
                        f"{100 / np.sqrt(2 * (n - 1)):.0f} %")
    elif sigma == "propagate" and S is not None:
        out_sigma, src = np.sqrt((S ** 2).sum(axis=0)) / n, s.sigma_source
    meta = dict(s.meta)
    meta["sweeps"] = {"averaged": n, "direction": direction, "grid": "interpolated" if interpolated else "identical"}
    out = Spectrum(X, mean, out_sigma, src, s.x_unit, s.y_unit, s.x_label, s.y_label, None, meta,
                   s.flags | ({"interpolated"} if interpolated else set()), {})
    for m in msgs:
        out = warn(out, m)
    if interpolated:
        out = warn(out, "Sweeps interpolated onto the grid of the first sweep (correlates neighbouring points).")
    return out
