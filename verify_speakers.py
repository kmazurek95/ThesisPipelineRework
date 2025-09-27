"""
Speaker Verification Tool

This script verifies if granules without entries in granule_members.csv
truly don't have speaker identification in their text.

## Example Command:
```powershell
python verify_speakers.py --normalized-dir data\normalized_114
```

## Arguments:
- `--normalized-dir`: Directory containing normalized data with by_package structure.
"""

import re
import os
import pandas as pd
from pathlib import Path
import logging
import argparse

# Import the speaker pattern used in speaker_attribution.py
# This is a simplified version that catches the most common speaker cues
SPEAKER_CUE_RE = re.compile(r"(?:^|\n)(?:Mr\.|Mrs\.|Ms\.|Hon\.|The\s+[A-Z]+(?:\s+[A-Z]+)*:)", re.MULTILINE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("speaker_verification")

def verify_speakers_in_text(normalized_dir: Path):
    """
    Verify if granules without entries in granule_members.csv truly don't have speaker identification in text.
    
    Args:
        normalized_dir: Path to the normalized data directory containing by_package/
    """
    by_pkg_dir = normalized_dir / "by_package"
    if not by_pkg_dir.exists():
        logger.error(f"Directory not found: {by_pkg_dir}")
        return
    
    # Track statistics
    stats = {
        "packages_checked": 0,
        "granules_without_members": 0,
        "granules_with_speaker_cues": 0,
        "false_negatives": []  # Granules with speaker cues but no members entries
    }
    
    # Process each package directory
    for pkg_dir in by_pkg_dir.glob("*"):
        if not pkg_dir.is_dir():
            continue
            
        stats["packages_checked"] += 1
        
        # Check if required files exist
        core_file = pkg_dir / "granules_core.csv"
        members_file = pkg_dir / "granule_members.csv"
        
        if not core_file.exists():
            logger.warning(f"No granules_core.csv found in {pkg_dir}")
            continue
            
        # Load granules core data
        granules_df = pd.read_csv(core_file)
        
        # Get list of granules with members (if members file exists)
        granules_with_members = set()
        if members_file.exists():
            members_df = pd.read_csv(members_file)
            granules_with_members = set(members_df["granuleId"].unique())
        
        # Identify granules without members
        all_granules = set(granules_df["granuleId"].unique())
        granules_without_members = all_granules - granules_with_members
        
        stats["granules_without_members"] += len(granules_without_members)
        
        # Check each granule without members for speaker cues in text
        for granule_id in granules_without_members:
            granule_row = granules_df[granules_df["granuleId"] == granule_id].iloc[0]
            
            # Get text content, prioritizing text_for_speaker
            text = ""
            for field in ["text_for_speaker", "text_readability", "parsed_text"]:
                if field in granule_row and isinstance(granule_row[field], str) and granule_row[field].strip():
                    text = granule_row[field]
                    break
            
            if not text:
                continue
                
            # Search for speaker cues in the text
            matches = list(SPEAKER_CUE_RE.finditer(text))
            
            if matches:
                stats["granules_with_speaker_cues"] += 1
                
                # This is a false negative: text has speaker cues but no members entry
                false_negative = {
                    "granuleId": granule_id,
                    "packageId": granule_row.get("packageId", ""),
                    "title": granule_row.get("title", ""),
                    "speaker_cues_found": len(matches),
                    "first_cue": matches[0].group(0) if matches else ""
                }
                stats["false_negatives"].append(false_negative)
                
                logger.warning(f"Found speaker cues in {granule_id} but no entry in granule_members.csv")
                
    # Report statistics
    logger.info(f"Packages checked: {stats['packages_checked']}")
    logger.info(f"Granules without members: {stats['granules_without_members']}")
    logger.info(f"Granules with speaker cues but no members: {stats['granules_with_speaker_cues']}")
    
    # Output details of false negatives
    if stats["false_negatives"]:
        logger.info("False negatives (granules with speaker cues but no members):")
        for i, fn in enumerate(stats["false_negatives"][:10]):  # Show first 10
            logger.info(f"{i+1}. {fn['granuleId']} - {fn['title']} - {fn['first_cue']}")
        
        # Save all false negatives to CSV
        false_neg_df = pd.DataFrame(stats["false_negatives"])
        output_file = normalized_dir / "speaker_verification_results.csv"
        false_neg_df.to_csv(output_file, index=False)
        logger.info(f"Saved all {len(stats['false_negatives'])} false negatives to {output_file}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Verify speaker identification in granules")
    parser.add_argument("--normalized-dir", required=True, help="Path to normalized data directory containing by_package/")
    args = parser.parse_args()
    
    normalized_dir = Path(args.normalized_dir)
    verify_speakers_in_text(normalized_dir)

if __name__ == "__main__":
    main()