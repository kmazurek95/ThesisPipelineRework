"""
Tests for data validation functionality.

These tests verify that the multi-level datasets are properly structured
and contain expected values.
"""

import pytest
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "output"


class TestDatasetStructure:
    """Test dataset structure and required columns."""

    @pytest.fixture
    def level1(self):
        """Load level 1 dataset."""
        gz_path = DATA_DIR / "level1.csv.gz"
        csv_path = DATA_DIR / "level1.csv"
        if gz_path.exists():
            return pd.read_csv(gz_path, nrows=1000)
        elif csv_path.exists():
            return pd.read_csv(csv_path, nrows=1000)
        else:
            pytest.skip("level1.csv not found - run pipeline first")

    @pytest.fixture
    def level2(self):
        """Load level 2 dataset."""
        path = DATA_DIR / "level2_org.csv"
        if not path.exists():
            pytest.skip("level2_org.csv not found - run pipeline first")
        return pd.read_csv(path)

    @pytest.fixture
    def level3(self):
        """Load level 3 dataset."""
        path = DATA_DIR / "level3_politician.csv"
        if not path.exists():
            pytest.skip("level3_politician.csv not found - run pipeline first")
        return pd.read_csv(path)

    @pytest.fixture
    def level4(self):
        """Load level 4 dataset."""
        path = DATA_DIR / "level4_policy.csv"
        if not path.exists():
            pytest.skip("level4_policy.csv not found - run pipeline first")
        return pd.read_csv(path)

    def test_level1_has_required_columns(self, level1):
        """Level 1 should have key columns."""
        required = ["org_id", "prominence_prediction"]
        for col in required:
            assert col in level1.columns, f"Missing column: {col}"

    def test_level1_prominence_is_binary(self, level1):
        """Prominence prediction should be 0 or 1."""
        unique_vals = level1["prominence_prediction"].dropna().unique()
        assert set(unique_vals).issubset({0, 1, 0.0, 1.0}), \
            f"Unexpected prominence values: {unique_vals}"

    def test_level2_has_required_columns(self, level2):
        """Level 2 should have aggregation columns."""
        required = ["org_id", "total_mentions", "avg_prominence"]
        for col in required:
            assert col in level2.columns, f"Missing column: {col}"

    def test_level2_mentions_positive(self, level2):
        """Mention counts should be positive."""
        assert (level2["total_mentions"] > 0).all(), \
            "All organizations should have at least 1 mention"

    def test_level2_prominence_in_range(self, level2):
        """Avg prominence should be between 0 and 1."""
        assert level2["avg_prominence"].between(0, 1).all(), \
            "Average prominence should be between 0 and 1"

    def test_level3_has_required_columns(self, level3):
        """Level 3 should have politician columns."""
        required = ["bioGuideId", "total_mentions", "party", "chamber"]
        for col in required:
            assert col in level3.columns, f"Missing column: {col}"

    def test_level3_party_values(self, level3):
        """Party should be D, R, or I."""
        valid_parties = {"D", "R", "I", None}
        actual_parties = set(level3["party"].dropna().unique())
        assert actual_parties.issubset(valid_parties), \
            f"Unexpected party values: {actual_parties - valid_parties}"

    def test_level3_chamber_values(self, level3):
        """Chamber should be H or S."""
        valid_chambers = {"H", "S", "House", "Senate", None}
        actual_chambers = set(level3["chamber"].dropna().unique())
        assert actual_chambers.issubset(valid_chambers), \
            f"Unexpected chamber values: {actual_chambers - valid_chambers}"

    def test_level4_has_required_columns(self, level4):
        """Level 4 should have policy area columns."""
        required = ["issue_area", "total_mentions"]
        for col in required:
            assert col in level4.columns, f"Missing column: {col}"

    def test_level4_has_multiple_areas(self, level4):
        """Should have multiple policy areas."""
        assert len(level4) > 1, "Should have multiple policy areas"


class TestDataConsistency:
    """Test cross-level data consistency."""

    @pytest.fixture
    def all_levels(self):
        """Load all datasets."""
        levels = {}
        for name, filename in [
            ("level2", "level2_org.csv"),
            ("level3", "level3_politician.csv"),
            ("level4", "level4_policy.csv"),
        ]:
            path = DATA_DIR / filename
            if path.exists():
                levels[name] = pd.read_csv(path)
            else:
                pytest.skip(f"{filename} not found")
        # level1 may be gzipped
        gz_path = DATA_DIR / "level1.csv.gz"
        csv_path = DATA_DIR / "level1.csv"
        if gz_path.exists():
            levels["level1"] = pd.read_csv(gz_path)
        elif csv_path.exists():
            levels["level1"] = pd.read_csv(csv_path)
        else:
            pytest.skip("level1.csv not found")
        return levels

    def test_level1_rows_match_aggregation(self, all_levels):
        """Level 1 row count should match sum of Level 2 mentions."""
        level1_count = len(all_levels["level1"])
        level2_sum = all_levels["level2"]["total_mentions"].sum()
        assert abs(level1_count - level2_sum) < 10, \
            f"Mismatch: Level 1 has {level1_count}, Level 2 sums to {level2_sum}"

    def test_unique_orgs_match(self, all_levels):
        """Unique orgs in Level 1 should match Level 2 count."""
        level1_orgs = all_levels["level1"]["org_id"].nunique()
        level2_orgs = len(all_levels["level2"])
        assert level1_orgs == level2_orgs, \
            f"Org count mismatch: {level1_orgs} vs {level2_orgs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
