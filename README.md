Here’s a tightened, clearer README you can drop in right now. I also added a short, dated “Project status” block at the top so visitors immediately know what’s in progress and what’s stable.

---

# Interest-Group Analysis Toolkit

> A modular, **reproducible** pipeline for collecting, processing, and analyzing U.S. Congressional text and related signals (GovInfo transcripts, Congress APIs, Google Trends). The toolkit normalizes raw inputs, extracts **interest-group** mentions with strict matching, and produces clean outputs for downstream modeling (e.g., prominence classification, aggregation, dashboards).

## Project status (Sept 29, 2025)

* **Revamp in progress:** speaker attribution and prominence **classification** are being refactored for reliability and speed.
* **Stable:** data collection, strict mention extraction, and sample fixtures for quick demos.
* **Performance note:** speaker attribution is CPU-intensive on full runs (hours on a typical laptop).
* **Original thesis code:** see the archived repo used for the Master’s thesis results:
  `https://github.com/<your-username>/<old-repo-name>`

---

## Features

**Data Collection**

* Fetch Congressional Record transcripts (GovInfo API).
* Retrieve bill/session metadata (Congress APIs).
* Pull member info and auxiliary context.
* Gather policy salience signals (Google Trends).

**Data Processing**

* Normalize raw JSON/HTML into tidy, versioned CSV/Parquet.
* Strictly extract & deduplicate **interest-group** mentions (canonical names + acronyms only).
* Link mentions to congressional speakers (attribution pipeline).
* Post-process mentions (highlighting spans, rollups, exports).

**Analysis**

* Aggregate mention frequencies by org, chamber, party, date.
* Join to salience metrics by year/topic for modeling (e.g., prominence).

---

## Repository Structure

```
ThesisPipelineRework/
├─ README.md                     # You are here
├─ LICENSE
├─ pyproject.toml / requirements.txt
├─ .env.example                  # Template for API keys & settings
├─ .gitignore                    # Excludes raw/processed data
├─ scripts/                      # Entry-point scripts (collection/processing)
│  ├─ 1.collect_govinfo.py
│  └─ ...more
├─ interest_group_analysis/      # Project modules/pipelines
│  ├─ 1.data_collection/
│  │  └─ interest_group_prep.py  # Preps org list (canonical + acronym)
│  └─ 2.data_processing/
│     ├─ 3.mention_extraction.py # Strict mention extraction
│     └─ 3.attach_speakers.py    # Speaker attribution
├─ data/
│  ├─ README.md                  # What belongs in data/, how to obtain it
│  ├─ sample/                    # Small fixtures for tests/demos (checked in)
│  ├─ raw/                       # (ignored) API dumps & HTML/JSON
│  └─ processed/                 # (ignored) normalized CSVs, mentions
└─ results/
   ├─ .gitkeep
   └─ README.md                  # What outputs mean & how to reproduce
```

> `data/raw/` and `data/processed/` are **git-ignored** to keep the repo lean. Use `data/sample/` for tiny fixtures committed to version control.

---

## Installation

> Python **3.10+** recommended.

```bash
# clone
git clone https://github.com/<you>/ThesisPipelineRework.git
cd ThesisPipelineRework

# create & activate a virtual environment
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
# or, if using pyproject:
# pip install -e .
```

---

## Configuration

Copy the example env and add your tokens (GovInfo, Congress, etc.):

```bash
cp .env.example .env
# then edit .env
```

---

## Quick Start

### 1) Prepare Interest-Group List

**Input:** `data/Interest_groups_manually_validated.xlsx`
Required columns: `org_id`, `original_name_2`, `current_name_2`, `acronym_2`

The prep script:

* fills `current_name_2` where blank using `original_name_2`
* keeps only `org_id`, `interest_group`, `acronym`
* writes `data/interest_groups_list.csv`

**Windows PowerShell**

```powershell
Set-Location .\interest_group_analysis\1.data_collection
python .\interest_group_prep.py `
  --in  "..\..\data\Interest_groups_manually_validated.xlsx" `
  --out "..\..\data\interest_groups_list.csv"
```

### 2) Extract Mentions (strict: canonical + acronym only)

**Inputs:**

* Normalized Congressional text under `data/normalized_<run>/` **or**
* Raw JSON/JSONL under `data/raw/`

**Windows PowerShell (normalized mode)**

```powershell
Set-Location .\interest_group_analysis\2.data_processing
python .\3.mention_extraction.py normalized `
  --normalized-dir "..\..\data\normalized_114" `
  --interest-csv   "..\..\data\interest_groups_list.csv" `
  --out-dir        "..\..\data\processed\mentions_114" `
  --threads 6 --resume `
  --only-canonical-names `
  --strict-current-acronym
```

**Output:**
`data/processed/mentions_114/mentions.jsonl`
(each line: `org_id`, `interest_group`, `variation`, `is_acronym`, `sentence`, …)

### 3) Attach Speakers to Mentions

**Inputs:**

* Mention JSONL from step 2
* Normalized Congressional text with spans

```powershell
Set-Location .\interest_group_analysis\2.data_processing
python .\3.attach_speakers.py `
  --mentions-jsonl ..\..\data\processed\mentions_114\mentions.jsonl `
  --normalized-dir ..\..\data\normalized\normalized_114 `
  --out-dir ..\..\data\processed\mentions_and_speaker_114 `
  --save-csv `
  --qa-jsonl
```

**Note:** This step is CPU-intensive on full datasets.

**Outputs:**

* `mentions_with_speakers.jsonl` (main)
* `mentions_with_speakers.csv` (optional)
* `speaker_qc.jsonl` (optional QA)
* `processed_granules.jsonl` (resume state)

---

## Reproducibility

* **Deterministic extraction:** whole-phrase matches only (no fuzzy alt names), reducing false positives.
* **Versioned sample data:** tiny fixtures in `data/sample/` enable quick end-to-end demos.
* **Large artifacts ignored:** `.gitignore` excludes raw/processed blobs; provenance is preserved via scripts/configs.
* **Environment capture:** pin via `requirements.txt` (or `pyproject.toml`) for consistent installs.

---

## License

MIT (see `LICENSE`).

---

## Citation

If this toolkit informs academic work, please cite this repository and upstream data providers (GovInfo, Congress APIs, Google Trends).