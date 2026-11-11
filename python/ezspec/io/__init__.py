"""Data import. File formats are pluggable, see :mod:`ezspec.io.registry`."""

from .readers import FORMATS, read_bytes, read_file, read_spectra, sniff
from .registry import (Reader, file_dialog_filter, find_reader, get_reader, is_supported, list_readers,
                       register_reader, source_meta, supported_extensions, unregister_reader)
from .text import TableInfo, parse_table, read_table

__all__ = ["FORMATS", "Reader", "TableInfo", "file_dialog_filter", "find_reader", "get_reader", "is_supported",
           "list_readers", "parse_table", "read_bytes", "read_file", "read_spectra", "read_table",
           "register_reader", "sniff", "source_meta", "supported_extensions", "unregister_reader"]
