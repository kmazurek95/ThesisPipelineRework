"""
Policy Salience Module for ThesisPipelineRework

This module uses Google Trends data to measure the salience of various policy topics
over time. It processes the data and generates metrics that can be used to analyze
how different policy areas gain or lose public attention.

Module Structure:
----------------
- config.py: Configuration settings including policy topics and file paths
- data_loader.py: Utilities for loading and saving data from various formats
- trends_collector.py: Interface to Google Trends API with robust retry handling
- analyzer.py: Analysis functions for calculating policy salience metrics
- visualizer.py: Visualization tools for trends and salience data
- main.py: Main script to run the full pipeline

Usage:
------
To run the full pipeline:
```
python -m interest_group_analysis.1.data_collection.4.policy_salience.main \
    --output-dir ./data/policy_salience \
    --trends-file google_trends_data.csv \
    --salience-file salience_mapping.csv
```

For testing with limited data:
```
python -m interest_group_analysis.1.data_collection.4.policy_salience.main \
    --dates-limit 5 \
    --output-dir ./data/policy_salience_test
```

To skip Google Trends data collection and use cached data:
```
python -m interest_group_analysis.1.data_collection.4.policy_salience.main \
    --skip-trends
```
"""

from . import config
from .data_loader import DataLoader
from .trends_collector import TrendsCollector
from .analyzer import SalienceAnalyzer
from .visualizer import SalienceVisualizer
from .main import main

__all__ = [
    'config',
    'DataLoader',
    'TrendsCollector',
    'SalienceAnalyzer',
    'SalienceVisualizer',
    'main',
]