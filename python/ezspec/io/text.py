"""Delimited text / CSV import with automatic format detection.

Handles comma, semicolon, tab and whitespace delimiters, decimal commas
(German locale exports), header lines, comment lines and trailing text.
Every numeric row with the modal number of columns is used.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


@dataclass
class TableInfo:
    delimiter: str | None          # None = whitespace
    decimal: str                   # "." or ","
    header: list = field(default_factory=list)
    n_header_lines: int = 0
    data: np.ndarray | None = None  # (rows, cols)
    skipped: int = 0
    encoding: str = "utf-8"

    @property
    def n_cols(self) -> int:
        return 0 if self.data is None else self.data.shape[1]


def decode_bytes(raw: bytes) -> tuple:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and raw[:2] not in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text, enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _split(line: str, delimiter):
    if delimiter is None:
        return line.split()
    return [t.strip() for t in line.split(delimiter)]


def _is_num(tok: str) -> bool:
    return bool(_NUM.match(tok.strip()))


def _candidates(lines):
    """Score (delimiter, decimal) combinations by how many lines parse consistently."""
    best = None
    for delim in ("\t", ";", ",", None):
        for dec in (".", ","):
            if delim == "," and dec == ",":
                continue
            counts = []
            for ln in lines:
                toks = [t for t in _split(ln, delim)]
                if delim is not None:
                    toks = [t for t in toks if t != ""] if delim != "\t" else toks
                if not toks:
                    continue
                ok = all(_is_num(t) and (dec == "," or "," not in t) and (dec == "." or "." not in t)
                         for t in toks)
                if ok:
                    counts.append(len(toks))
            if not counts:
                continue
            vals, freq = np.unique(counts, return_counts=True)
            mode = int(vals[np.argmax(freq)])
            score = (int(freq.max()), mode)
            if mode >= 1 and (best is None or score > best[0]):
                best = (score, delim, dec, mode)
    return best


def parse_table(text: str, delimiter: str | None = "auto", decimal: str = "auto",
                comment: str = "#") -> TableInfo:
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    content = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(comment)]
    if delimiter == "auto" or decimal == "auto":
        sample = content[:2000]
        best = _candidates(sample)
        if best is None:
            raise ValueError("keine numerischen Datenzeilen gefunden")
        _, d_auto, dec_auto, _ = best
        delimiter = d_auto if delimiter == "auto" else delimiter
        decimal = dec_auto if decimal == "auto" else decimal
    rows = []
    header = []
    n_header = 0
    skipped = 0
    ncols = None
    for ln in content:
        toks = _split(ln, delimiter)
        if delimiter not in (None, "\t"):
            toks = [t for t in toks if t != ""]
        if toks and all(_is_num(t) for t in toks):
            vals = [float(t.replace(",", ".")) if decimal == "," else float(t) for t in toks]
            if ncols is None:
                ncols = len(vals)
            if len(vals) == ncols:
                rows.append(vals)
            else:
                skipped += 1
        elif not rows:
            n_header += 1
            header = next(csv.reader(io.StringIO(ln), delimiter=delimiter or " ", skipinitialspace=True))
            header = [h.strip() for h in header if h.strip()]
        else:
            skipped += 1
    if not rows:
        raise ValueError("keine numerischen Datenzeilen gefunden")
    data = np.array(rows, float)
    if len(header) != data.shape[1]:
        header = [f"Spalte {i + 1}" for i in range(data.shape[1])]
    return TableInfo(delimiter, decimal, header, n_header, data, skipped)


def read_table(path, **kw) -> TableInfo:
    raw = Path(path).read_bytes()
    text, enc = decode_bytes(raw)
    info = parse_table(text, **kw)
    info.encoding = enc
    return info


def guess_unit(name: str) -> str:
    n = name.lower().replace(" ", "")
    if "cm-1" in n or "cm^-1" in n or "cm⁻¹" in n or "1/cm" in n or "wavenumber" in n or "wellenzahl" in n:
        return "raman" if "raman" in n or "shift" in n else "cm-1"
    if "(nm)" in n or "[nm]" in n or "/nm" in n or "wavelength" in n or "wellenlänge" in n:
        return "nm"
    if "(ev)" in n or "[ev]" in n or "/ev" in n or "energy" in n or "energie" in n:
        return "eV"
    return ""
