"""Safe user formulas with an arbitrary number of parameters.

Example::

    f = Formula("y0 + A*exp(-x/tau)")
    f.parameters            # ('y0', 'A', 'tau')
    f(x, y0=1, A=2, tau=3)

Syntax is Python/NumPy arithmetic; ``^`` is accepted as power and ``ln`` as
natural log. Every free name that is neither an independent variable, a
function nor ``pi`` is a fit parameter. The expression is parsed into an AST
and checked against a whitelist (no attribute access, subscripts, lambdas,
comprehensions or names starting with ``_``) before it is compiled, so
arbitrary code cannot be executed.

Lineshapes are available inside formulas, e.g.
``gauss(x, A1, c1, w1) + gauss(x, A2, c2, w2) + b``.
"""

from __future__ import annotations

import ast
import re

import numpy as np
from scipy import special

from .. import lineshapes as ls


def _where(cond, a, b):
    return np.where(cond, a, b)


FUNCTIONS = {
    # elementary
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos,
    "arctan": np.arctan, "arctan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "minimum": np.minimum, "maximum": np.maximum, "heaviside": np.heaviside, "where": _where,
    "sinc": np.sinc, "expm1": np.expm1, "log1p": np.log1p,
    # special functions
    "erf": special.erf, "erfc": special.erfc, "erfcx": special.erfcx, "gamma": special.gamma,
    "gammaln": special.gammaln, "besselj": special.jv, "expit": special.expit,
    # lineshapes (area-normalised, FWHM widths)
    "gauss": ls.gaussian, "gaussian": ls.gaussian, "lorentz": ls.lorentzian,
    "lorentzian": ls.lorentzian, "voigt": ls.voigt, "pvoigt": ls.pseudo_voigt,
    "tch": ls.tch_pseudo_voigt, "pearson7": ls.pearson7, "emg": ls.emg,
}
CONSTANTS = {"pi": np.pi}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class FormulaError(ValueError):
    pass


def preprocess(expr: str) -> str:
    expr = expr.strip()
    if "=" in expr and not re.search(r"[<>!=]=", expr):
        # allow "y = ..." / "f(x) = ..."
        expr = expr.split("=", 1)[1].strip()
    return expr.replace("^", "**")


class Formula:
    def __init__(self, expression: str, independent=("x",)):
        self.expression = expression
        self.independent = tuple(independent)
        src = preprocess(expression)
        if not src:
            raise FormulaError("leerer Ausdruck")
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"Syntaxfehler: {exc.msg} (Spalte {exc.offset})") from None
        names = []
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise FormulaError(f"nicht erlaubtes Element: {type(node).__name__}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise FormulaError("nur Zahlen-Konstanten erlaubt")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    fname = getattr(node.func, "id", "?")
                    raise FormulaError(f"unbekannte Funktion: {fname}")
                if node.keywords:
                    raise FormulaError("Schlüsselwort-Argumente sind nicht erlaubt")
            if isinstance(node, ast.Name):
                if node.id.startswith("_"):
                    raise FormulaError(f"Name {node.id!r} nicht erlaubt")
                names.append((node.col_offset, node.id))
        names = [n for _, n in sorted(names)]   # order of appearance in the expression
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
        params = []
        for n in names:
            if n in self.independent or n in CONSTANTS or n in called or n in params:
                continue
            if n in FUNCTIONS:
                raise FormulaError(f"{n!r} ist ein Funktionsname und kann kein Parameter sein")
            params.append(n)
        missing = [v for v in self.independent if v not in names]
        if missing and len(self.independent) == 1:
            raise FormulaError(f"der Ausdruck hängt nicht von {missing[0]!r} ab")
        self.parameters = tuple(params)
        self._code = compile(tree, "<formula>", "eval")
        self._src = src

    def __call__(self, *xs, **params):
        if len(xs) != len(self.independent):
            raise TypeError(f"expected {len(self.independent)} independent variable(s)")
        ns = dict(FUNCTIONS)
        ns.update(CONSTANTS)
        ns.update(zip(self.independent, xs))
        missing = [p for p in self.parameters if p not in params]
        if missing:
            raise TypeError(f"missing parameter(s): {missing}")
        ns.update({p: params[p] for p in self.parameters})
        with np.errstate(all="ignore"):
            out = eval(self._code, {"__builtins__": {}}, ns)  # noqa: S307 - AST whitelisted above
        return np.broadcast_to(np.asarray(out, float), np.shape(xs[0])).copy()

    def __repr__(self):
        return f"Formula({self.expression!r}, params={self.parameters})"


def variable_name(label: str, taken=()) -> str:
    """A valid, non-reserved formula identifier from a column header,
    e.g. 'T / °C' -> 'T', 'laser temp. (K)' -> 'laser_temp'."""
    import keyword
    import re
    base = re.split(r"\s*[/\[(]", str(label).strip(), maxsplit=1)[0]
    name = re.sub(r"\W+", "_", base, flags=re.ASCII).strip("_") or "v"
    if name[0].isdigit():
        name = "v_" + name
    if name in FUNCTIONS or name in CONSTANTS or name == "x" or keyword.iskeyword(name):
        name = name + "_v"
    out, i = name, 2
    while out in taken:
        out = f"{name}{i}"
        i += 1
    return out
