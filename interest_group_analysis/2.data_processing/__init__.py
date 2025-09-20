"""
Subpackage for cleaning and transforming raw data.

Each module in this package handles a specific aspect of the data
processing workflow, such as parsing API responses, assigning
speaker information, extracting interest group mentions, expanding
nested structures, and providing helper utilities.
"""

__all__ = [
    "api_results",
    "speaker_assignment",
    "mention_extraction",
    "expander",
    "utils",
]
