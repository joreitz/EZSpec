"""Number formatting for reports, tables and figures.

Values with an uncertainty are rounded to a chosen number of significant
digits *of the uncertainty*, and the value to the same decimal place (as
recommended by the GUM, JCGM 100:2008, 7.2.6). The default is 2 digits; it
can be raised (up to 15) or rounding switched off (``unc_digits = 0``).
Values without uncertainty use ``digits`` significant digits.

Rounding only affects the display. Exported tables (CSV/JSON) always contain
the full double-precision values.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

_SETTINGS = {"unc_digits": 2, "digits": 6}


def set_precision(unc_digits: int | None = None, digits: int | None = None) -> None:
    """Set the display precision. ``unc_digits``: significant digits of the
    uncertainty (1–15; 0 = no rounding); ``digits``: significant digits of
    values without uncertainty (1–17)."""
    if unc_digits is not None:
        u = int(unc_digits)
        if not 0 <= u <= 15:
            raise ValueError("unc_digits must be between 0 (no rounding) and 15")
        _SETTINGS["unc_digits"] = u
    if digits is not None:
        d = int(digits)
        if not 1 <= d <= 17:
            raise ValueError("digits must be between 1 and 17")
        _SETTINGS["digits"] = d


def get_precision() -> dict:
    return dict(_SETTINGS)


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(v)
    except TypeError:
        return False


def fmt_num(v, digits: int | None = None) -> str:
    """Value without uncertainty with ``digits`` significant digits."""
    if v is None:
        return ""
    if not _finite(v):
        return str(v)
    n = _SETTINGS["digits"] if digits is None else digits
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-4 <= a < 10 ** max(n, 6):
        return f"{v:.{n}g}"
    return f"{v:.{max(n - 1, 0)}e}"


def fmt_value(v, e, digits: int | None = None) -> str:
    """'value ± uncertainty', rounded to ``digits`` significant digits of the
    uncertainty (default: the global setting; 0 = no rounding)."""
    if v is None or not _finite(v):
        return str(v)
    if e is None or not _finite(e) or e <= 0:
        return fmt_num(v)
    n = _SETTINGS["unc_digits"] if digits is None else digits
    if n <= 0:
        return f"{v:.15g} ± {e:.15g}"
    q = math.floor(math.log10(e)) - (n - 1)          # decimal position of the last digit kept
    if _round(e, q) >= Decimal(10) ** (q + n):        # rounding carried into the next decade (0.96 → 1.0)
        q += 1
    mag = math.floor(math.log10(max(abs(v), e)))
    if -4 <= mag < 7:
        return f"{_fixed(v, q)} ± {_fixed(e, q)}"
    # common power of ten for very large or small numbers
    return f"({_fixed(_shift(v, -mag), q - mag)} ± {_fixed(_shift(e, -mag), q - mag)})e{mag:+03d}"


def _round(v, q) -> Decimal:
    """``v`` rounded half-up at decimal position 10**q (on its shortest decimal representation)."""
    d = v if isinstance(v, Decimal) else Decimal(repr(float(v)))
    return d.quantize(Decimal(1).scaleb(q), rounding=ROUND_HALF_UP)


def _fixed(v, q) -> str:
    r = _round(v, q)
    return f"{r:.{max(-q, 0)}f}"


def _shift(v, k) -> Decimal:
    return Decimal(repr(float(v))).scaleb(k)
