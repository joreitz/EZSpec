"""EZSpec – interactive spectrum processing and curve fitting with honest statistics.

Layers: Rust numerical core (``ezspec._core``, optional, with NumPy/SciPy
reference fallback), Python domain layer (``Spectrum``, ``ops``,
``Pipeline``, ``models``, ``fit``), Qt GUI (``ezspec.gui``) and export
(``ezspec.export``).
"""

__version__ = "0.1.0"

from . import ops  # noqa: E402
from ._backend import HAVE_RUST, backend_name, set_backend  # noqa: E402
from .fit import FitOptions, compare, fit, fit_xy  # noqa: E402
from .models import Model, add_peak, add_template, find_peaks  # noqa: E402
from .pipeline import Pipeline  # noqa: E402
from .spectrum import SigmaSource, Spectrum, spectrum  # noqa: E402

__all__ = ["FitOptions", "HAVE_RUST", "Model", "Pipeline", "SigmaSource", "Spectrum", "add_peak",
           "add_template", "backend_name", "compare", "find_peaks", "fit", "fit_xy", "ops", "set_backend",
           "spectrum", "__version__"]
