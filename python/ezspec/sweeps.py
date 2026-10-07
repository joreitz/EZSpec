"""Back-and-forth measurements: detect sweeps (ramps) in acquisition order.

Data such as a TDLAS current ramp run x up and down, so one x value has
several y values. The readers store the acquisition order in
``aux["acq_index"]``; spectra themselves are kept sorted by x.

Turning points are found with a hysteresis threshold: the direction only
changes once x has moved back from the running extreme by more than
``tolerance`` (a fraction of the total x range). Noise in a measured x
(e.g. a read-back current) therefore does not split a ramp. Segments whose
x span is below the threshold (plateaus, laser-off periods) are classified
as flat.

Up and down sweeps are kept apart deliberately: with a current-tuned laser
the wavelength at a given current generally differs between the two
directions (thermal lag), so averaging them would mix different x scales.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

_MV_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


@dataclass
class Sweep:
    indices: np.ndarray     # indices into the spectrum arrays, in acquisition order
    direction: int          # +1 up, -1 down, 0 flat
    number: int             # running number among sweeps of the same direction

    @property
    def n(self) -> int:
        return len(self.indices)


def acquisition_order(s) -> np.ndarray | None:
    a = s.aux.get("acq_index")
    if a is None:
        return None
    return np.argsort(np.asarray(a), kind="stable")


def find_sweeps(s, tolerance: float = 0.01, min_points: int = 5) -> list:
    """Sweeps of a spectrum in acquisition order (requires ``aux['acq_index']``)."""
    order = acquisition_order(s)
    if order is None:
        raise ValueError("Aufnahmereihenfolge unbekannt – Datei neu importieren")
    xs = s.x[order]
    n = len(xs)
    if n < 2:
        return []
    span = float(np.ptp(xs))
    tol = tolerance * span if span > 0 else 0.0
    turns = [0]
    direction = 0
    ext = 0
    for i in range(1, n):
        if direction == 0:
            if xs[i] - xs[0] > tol:
                direction, ext = 1, i
            elif xs[0] - xs[i] > tol:
                direction, ext = -1, i
        elif direction == 1:            # strict: a held extreme starts the next sweep
            if xs[i] > xs[ext]:
                ext = i
            elif xs[ext] - xs[i] > tol:
                turns.append(ext)
                direction, ext = -1, i
        else:
            if xs[i] < xs[ext]:
                ext = i
            elif xs[i] - xs[ext] > tol:
                turns.append(ext)
                direction, ext = 1, i
    if turns[-1] != n - 1:
        turns.append(n - 1)
    sweeps = []
    counters = {1: 0, -1: 0, 0: 0}
    start = 0
    for end in turns[1:]:
        seg = np.arange(start, end + 1)
        start = end + 1
        if len(seg) == 0:
            continue
        seg_span = float(np.ptp(xs[seg])) if len(seg) > 1 else 0.0
        if seg_span <= tol or len(seg) < 2:
            d = 0
        else:
            d = 1 if xs[seg[-1]] > xs[seg[0]] else -1
        if d != 0 and len(seg) < min_points:
            d = 0           # e.g. the fast fly-back of a sawtooth
        sweeps.append(Sweep(order[seg], d, counters[d]))
        counters[d] += 1
    return sweeps


def count_sweeps(s, tolerance: float = 0.01, min_points: int = 5) -> dict:
    sw = find_sweeps(s, tolerance, min_points)
    return {"up": sum(1 for w in sw if w.direction == 1), "down": sum(1 for w in sw if w.direction == -1),
            "flat": sum(1 for w in sw if w.direction == 0)}


def is_multivalued(s, tolerance: float = 0.01, min_points: int = 5) -> bool:
    """True if the data contain more than one non-flat sweep (x is not single-valued)."""
    if "acq_index" not in s.aux or s.n < 4:
        return False
    default = tolerance == 0.01 and min_points == 5
    if default:
        try:
            return _MV_CACHE[s]
        except (KeyError, TypeError):
            pass
    try:
        c = count_sweeps(s, tolerance, min_points)
        out = c["up"] + c["down"] > 1
    except ValueError:
        out = False
    if default:
        try:
            _MV_CACHE[s] = out
        except TypeError:
            pass
    return out


def display_order(s) -> np.ndarray | None:
    """Order for drawing: acquisition order for multi-valued data, else None (x order)."""
    return acquisition_order(s) if is_multivalued(s) else None


def display_order_masked(s, mask) -> np.ndarray | None:
    """Like :func:`display_order` for arrays that hold only the points ``s[mask]``
    (e.g. fit residuals)."""
    if not is_multivalued(s):
        return None
    acq = np.asarray(s.aux["acq_index"])[np.asarray(mask, bool)]
    return np.argsort(acq, kind="stable")
