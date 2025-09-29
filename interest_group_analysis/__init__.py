"""
Interest Group Analysis Package

This package provides modular components for data collection, processing,
classification, integration, and analysis of interest group prominence
in legislative debates. The pipeline is organized in sequential stages:

1. Data Collection: Fetch raw data from GovInfo, Congress.gov, and other sources
2. Data Processing: Clean, normalize, and extract structured information
3. Classification: Identify and categorize interest group mentions
4. Integration: Link mentions to bills, members, and policy areas
5. Analysis: Analyze prominence patterns and generate visualizations

Use the functions in `pipelines` module to orchestrate full workflow stages,
or import individual modules for targeted tasks.
"""

from . import config  # noqa: F401
from . import pipelines  # noqa: F401

__all__ = [
    "config",
    "pipelines",
]
