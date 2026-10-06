"""File readers returning :class:`~ezspec.spectrum.Spectrum` objects."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..spectrum import SigmaSource, Spectrum
from .jcamp import map_x_unit, parse_jcamp
from .text import decode_bytes, guess_unit, parse_table

FORMATS = {
    "text": "Text/CSV (*.txt *.csv *.dat *.tsv *.xy *.asc *.prn)",
    "jcamp": "JCAMP-DX (*.jdx *.dx *.jcm)",
}
_JCAMP_EXT = {".jdx", ".dx", ".jcm", ".jcamp"}


def _source(path: Path, raw: bytes, reader: str, options: dict) -> dict:
    return {"filename": path.name, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "reader": reader, "options": options}


def sniff(path) -> dict:
    """Detect format and, for text, the table layout (used by the import dialog)."""
    path = Path(path)
    raw = path.read_bytes()
    text, enc = decode_bytes(raw)
    if path.suffix.lower() in _JCAMP_EXT or text.lstrip().startswith("##"):
        return {"format": "jcamp", "encoding": enc}
    info = parse_table(text)
    return {"format": "text", "encoding": enc, "table": info}


def read_spectra(path, x_col: int = 0, y_cols=None, sigma_col: int | None = None, delimiter="auto",
                 decimal="auto", x_unit: str | None = None, extra_cols: dict | None = None) -> list:
    """Read one or several spectra. For text tables, every column in ``y_cols``
    (default: all columns except x and sigma) becomes one spectrum.
    ``extra_cols`` maps a variable name to a column used as an additional
    independent variable (stored as ``aux['var:<name>']``)."""
    path = Path(path)
    raw = path.read_bytes()
    text, _ = decode_bytes(raw)
    if path.suffix.lower() in _JCAMP_EXT or text.lstrip().startswith("##"):
        d = parse_jcamp(text)
        unit = map_x_unit(d["x_units"])
        x = d["x"]
        if unit == "um":
            x, unit = x * 1000.0, "nm"
        meta = {"name": d["title"] or path.stem, "source": _source(path, raw, "jcamp", {}),
                "jcamp": {k: v for k, v in d["labels"].items() if len(v) < 200},
                "import_warnings": d["warnings"]}
        s = Spectrum(x, d["y"], x_unit=x_unit or unit, y_unit=d["y_units"].lower(),
                     x_label=d["x_units"] or "x", y_label=d["y_units"] or "y", meta=meta).sorted()
        return [s]
    info = parse_table(text, delimiter=delimiter, decimal=decimal)
    data = info.data
    ncol = data.shape[1]
    if ncol == 1:
        data = np.column_stack([np.arange(len(data), dtype=float), data[:, 0]])
        info.header = ["Index"] + info.header
        ncol = 2
    if y_cols is None:
        skip = {x_col} | ({sigma_col} if sigma_col is not None else set()) | set((extra_cols or {}).values())
        y_cols = [c for c in range(ncol) if c not in skip]
    out = []
    xname = info.header[x_col]
    for c in y_cols:
        opts = {"x_col": x_col, "y_col": c, "sigma_col": sigma_col, "delimiter": info.delimiter,
                "decimal": info.decimal, "extra_cols": extra_cols}
        sigma = data[:, sigma_col] if sigma_col is not None else None
        aux = {f"var:{k}": data[:, v] for k, v in (extra_cols or {}).items()}
        good = np.isfinite(data[:, x_col]) & np.isfinite(data[:, c])
        if sigma is not None:
            good &= np.isfinite(sigma) & (sigma > 0)
        name = info.header[c] if len(y_cols) > 1 else path.stem
        meta = {"name": name, "source": _source(path, raw, "text", opts), "columns": info.header}
        s = Spectrum(data[good, x_col], data[good, c], None if sigma is None else sigma[good],
                     SigmaSource.KNOWN if sigma is not None else SigmaSource.UNKNOWN,
                     x_unit=x_unit if x_unit is not None else guess_unit(xname), x_label=xname,
                     y_label=info.header[c], meta=meta,
                     aux={k: v[good] for k, v in aux.items()}).sorted()
        out.append(s)
    return out


def read_file(path, index: int = 0, **kw) -> Spectrum:
    """Read a single spectrum (``index`` selects among several y columns)."""
    return read_spectra(path, **kw)[index]
