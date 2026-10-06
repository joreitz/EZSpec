"""Fit models: peaks, baseline terms, classic functions and free formulas."""

from .components import COMPONENT_TYPES, Component, ComponentType, ParamSetting
from .formula import FUNCTIONS, Formula, FormulaError
from .library import TEMPLATES, add_peak, add_template, area_from_height, find_peaks
from .model import Model, ModelError

__all__ = ["COMPONENT_TYPES", "Component", "ComponentType", "FUNCTIONS", "Formula", "FormulaError", "Model",
           "ModelError", "ParamSetting", "TEMPLATES", "add_peak", "add_template", "area_from_height",
           "find_peaks"]
