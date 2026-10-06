"""Versioned processing operations. Each takes a Spectrum and returns a new one.

>>> from ezspec import ops
>>> s = ops.crop(s, xmin=200, xmax=1800)
>>> s = ops.baseline_arpls(s, lam=1e6)
"""

from .registry import OpCall, OpSpec, ParamSpec, apply_op, get_op, list_ops, pop_warnings, warn
from .basic import (convert_x, crop, despike, estimate_noise, exclude, normalize, offset_scale,
                    resample, set_sigma, smooth_moving_average, smooth_savgol, smooth_whittaker)
from .baseline import (baseline_anchors, baseline_arpls, baseline_asls, baseline_polynomial,
                       baseline_rubberband, baseline_snip, whittaker_cutoff_points)

__all__ = [
    "OpCall", "OpSpec", "ParamSpec", "apply_op", "get_op", "list_ops", "pop_warnings", "warn",
    "convert_x", "crop", "despike", "estimate_noise", "exclude", "normalize", "offset_scale",
    "resample", "set_sigma", "smooth_moving_average", "smooth_savgol", "smooth_whittaker",
    "baseline_anchors", "baseline_arpls", "baseline_asls", "baseline_polynomial",
    "baseline_rubberband", "baseline_snip", "whittaker_cutoff_points",
]
