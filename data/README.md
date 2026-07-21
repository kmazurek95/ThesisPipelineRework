# Data directory

This directory contains all data files for the Interest Group Analysis pipeline.

## Directory structure

```
data/
├── reference/          # Static reference files (interest group lists, WRS)
├── training/           # ML training data (labeled mentions)
├── raw/                # Raw data from APIs/downloads
├── intermediate/       # Pipeline intermediate outputs
└── output/             # Final analysis-ready datasets
```

## Subdirectories

### reference/
Static reference files that don't change during pipeline runs:
- `interest_groups_list.csv` - Master list of interest groups with org_id
- `interest_group_names_and_acronyms.csv` - Name/acronym mappings
- `interest_groups_manually_validated.xlsx` - Manual validation work
- `washington_representatives_study.rda` - WRS lobbying/organizational metadata
- `washington_representatives_study.pdf` - WRS documentation

### training/
Machine learning training data:
- `combined_labeled.csv` - Combined labeled training data for prominence classifier
- `labeling/` - Batched labeling exports for manual annotation

### raw/
Raw data collected from external sources:
- `crec_114/` - Congressional Record XML files (114th Congress)
- `crec_115/` - Congressional Record XML files (115th Congress)
- `bills/` - Bill metadata from Congress.gov API
- `members/` - Congress member biographical data

### intermediate/
Pipeline intermediate outputs (can be regenerated):
- `normalized_114/`, `normalized_115/` - Parsed and normalized CREC data
  - `by_package/` - Per-date package folders with granules, committees, members, references
- `mentions_114/`, `mentions_115/` - Extracted interest group mentions
  - `mentions.jsonl` - Raw extracted mentions
  - `labeled_mentions.jsonl` - Mentions with prominence labels
  - `labeled_mentions.csv` - CSV format of labeled mentions
- `mentions_with_speakers/` - Mentions with speaker attribution attached

### output/
Final analysis-ready datasets:
- `level1.csv.gz` - Individual mention level (base unit of analysis, gzip compressed)
- `level2_org.csv` - Organization-level aggregation
- `level3_politician.csv` - Politician-level aggregation
- `level4_policy.csv` - Policy area-level aggregation
- `issue_salience_long.csv` - Google Trends policy salience data
- `multi_level_data.csv` - Legacy combined dataset (historical reference)

## Data flow

```
raw/crec_114/, crec_115/  →  intermediate/normalized_114/, normalized_115/
                          →  intermediate/mentions_114/, mentions_115/
                          →  intermediate/mentions_with_speakers/
                          →  output/level1.csv.gz
                          →  output/level2_org.csv (aggregated)
                          →  output/level3_politician.csv (aggregated)
                          →  output/level4_policy.csv (aggregated)
```

## Regenerating data

1. Raw data: Run `python -m interest_group_analysis.1.data_collection.1.govinfo`
2. Normalized: Run `python -m interest_group_analysis.2.data_processing.1.process_and_normalize`
3. Mentions: Run `python -m interest_group_analysis.2.data_processing.2.mention_extraction`
4. Speakers: Run `python -m interest_group_analysis.2.data_processing.3.attach_speakers`
5. Labels: Run `python -m interest_group_analysis.3.classification.classify_mentions`
6. Output: Run `python -m interest_group_analysis.4_integration.build_analysis_dataset`

## Notes

- Large data files (raw/, intermediate/) are gitignored
- Only reference/, training/, and small output files should be committed
- See `.gitignore` for specific exclusion patterns
