"""
Interest Group Analysis Package

This package provides modular components for data collection, processing,
classification, integration, and analysis of interest group prominence
in legislative debates. The pipeline is organized in sequential stages:

1. Data Collection (1.data_collection/): Fetch raw data from GovInfo, Congress.gov
2. Data Processing (2.data_processing/): Clean, normalize, and extract mentions
3. Classification (3.classification/): ML-based prominence classification
4. Integration (4_integration/): Data merging and feature engineering
5. Analysis (5_analysis/): Statistical analysis and visualization

Usage:
    Most modules are designed to run as standalone scripts:

    # Run the full integration pipeline
    python -m interest_group_analysis.4_integration.build_analysis_dataset

    # Run regression analysis
    python -m interest_group_analysis.5_analysis.regression_analysis

    # Train the classifier
    python -m interest_group_analysis.3.classification.text_classifier

    For orchestrated pipelines, use scripts/run_pipeline.py:

    python scripts/run_pipeline.py --stage integrate
    python scripts/run_pipeline.py --stage analyze

Note:
    The numbered folder prefixes (1., 2., etc.) indicate pipeline order but
    make direct Python imports challenging. Use the command-line interfaces
    or the scripts/ directory for running pipeline stages.
"""

from . import config  # noqa: F401

__version__ = "2.0.0"

__all__ = [
    "config",
    "__version__",
]
