How to retrieve data and what lands here.
# Data Directory

This directory contains data files used in the project. Below is an overview of the subdirectories and their purposes:

## Structure
- `raw/`: Contains raw, unprocessed data. **Do not commit large or sensitive files to the repository.**
- `processed/`: Contains processed data that has been cleaned or transformed for analysis.
- `sample/`: Contains small sample datasets for testing and demonstration purposes.

## Guidelines
1. **Raw Data**:
   - Raw data files should not be committed to the repository if they are large or sensitive.
   - Instead, provide instructions for downloading or generating the raw data.

2. **Processed Data**:
   - Processed data files can be committed if they are essential for reproducibility.
   - Ensure that any transformations are documented in the code.

3. **Sample Data**:
   - Include small, anonymized datasets that can be used to test the pipeline.
   - These files should be lightweight and easy to understand.

## How to Retrieve Data
- Raw data can be downloaded from [source name or URL].
- Alternatively, run the `scripts/1.collect_govinfo.py` script to collect the necessary data.
