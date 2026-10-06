"""Fitting with explicit statistics semantics."""

from .batch import GlobalModel, SeriesResult, fit_global, fit_series, split_global
from .compare import ComparisonError, compare
from .engine import METHODS, FitError, FitOptions, fit, fit_xy, propagate
from .result import FitResult, FitWarning, fmt_value
from .uncertainty import bootstrap, profile_ci

__all__ = ["METHODS", "ComparisonError", "FitError", "FitOptions", "FitResult", "FitWarning", "GlobalModel",
           "SeriesResult", "bootstrap", "compare", "fit", "fit_global", "fit_series", "fit_xy", "fmt_value",
           "profile_ci", "propagate", "split_global"]
