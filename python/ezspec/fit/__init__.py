"""Fitting with explicit statistics semantics."""

from .batch import GlobalModel, SeriesResult, fit_global, fit_series, split_global
from .compare import ComparisonError, compare, simulate_nested_test
from .engine import METHODS, FitError, FitOptions, fit, fit_xy, propagate
from .result import FitResult, FitWarning, fmt_value
from .systematics import baseline_systematics
from .uncertainty import bootstrap, mcmc, profile_ci

__all__ = ["METHODS", "ComparisonError", "FitError", "FitOptions", "FitResult", "FitWarning", "GlobalModel",
           "SeriesResult", "baseline_systematics", "bootstrap", "compare", "fit", "fit_global", "fit_series", "fit_xy", "fmt_value", "mcmc",
           "profile_ci", "propagate", "simulate_nested_test", "split_global"]
