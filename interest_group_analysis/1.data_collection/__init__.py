"""
Subpackage for collecting raw data from external APIs.

Modules in this package define functions that interact with public
APIs such as GovInfo, the Congress API, and Google Trends.  Each
function accepts an output directory where results should be
persisted.
"""

__all__ = [
    "govinfo",
    "bill_metadata",
    "congress_members",
    "policy_salience",
]
