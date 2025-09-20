"""
Interest Group Analysis Package

This package provides modular components for data collection, processing,
classification, integration, and analysis of interest group prominence
in legislative debates.  Modules are organised by stage and can be
used independently or orchestrated together through the high‑level
pipeline functions.
"""

from . import config  # noqa: F401
from . import pipelines  # noqa: F401

__all__ = [
    "config",
    "pipelines",
]
