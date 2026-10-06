"""Data import."""

from .readers import FORMATS, read_file, read_spectra, sniff
from .text import TableInfo, parse_table, read_table

__all__ = ["FORMATS", "TableInfo", "parse_table", "read_file", "read_spectra", "read_table", "sniff"]
