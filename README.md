# Legislative Data Toolkit

A modular, **reproducible** pipeline for collecting, processing, and analyzing U.S. legislative text and related signals (GovInfo transcripts, Congress APIs, Google Trends). The toolkit normalizes raw inputs, extracts **interest group** mentions with strict matching, and produces clean outputs for downstream analysis.

> **Project status**
>
> The **classification, integration, and analysis** rework is **in progress** and not yet complete.  
> For the original codebase used to produce the Master’s thesis results, please see the **old repository**:  
> https://github.com/<your-username>/<old-repo-name>

---

## Features

### Data Collection
- Fetch Congressional Record transcripts (GovInfo API).
- Retrieve bill/session metadata (Congress APIs).
- Pull member info and auxiliary context.
- Gather policy salience signals (Google Trends).

### Data Processing
- Normalize raw JSON/HTML into tidy CSVs.
- Strictly extract & deduplicate **interest group** mentions (name + acronym only).
- Post-process mentions (highlighting, rollups, exports).

### Analysis
- Aggregate mention frequencies by org, chamber, party, date.
- Join to salience metrics by year/topic for modeling.

---

## Repository Structure

```

ThesisPipelineRework/
├── README.md                 # You are here
├── LICENSE
├── pyproject.toml / requirements.txt
├── .env.example              # Template for API keys & settings
├── .gitignore
├── scripts/                  # Executable scripts (collection/processing)
│   ├── 1.collect\_govinfo.py
│   └── ...more
├── interest\_group\_analysis/  # Project-specific modules/pipelines
│   ├── 1.data\_collection/
│   │   └── interest\_group\_prep.py   # Preps org list (canonical + acronym)
│   └── 2.data\_processing/
│       └── 3.mention\_extraction.py  # Strict mention extraction
├── data/
│   ├── .gitkeep
│   ├── README.md             # What belongs in data/, how to obtain it
│   ├── sample/               # Small, versioned fixtures for tests/demos
│   ├── raw/                  # (ignored) API dumps & HTML/JSON
│   └── processed/            # (ignored) normalized CSVs, mentions
└── results/
├── .gitkeep
└── README.md             # What outputs mean & how to reproduce

````

> `data/raw/` and `data/processed/` are **ignored** by Git to keep the repo light. Use `data/sample/` for tiny fixtures checked into version control.

---

## Installation

> Python 3.10+ recommended.

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

# install deps
pip install -r requirements.txt
````

---

## Configuration

Copy the example env file and fill in keys (GovInfo, Congress, etc.):

```bash
cp .env.example .env
# edit .env with your tokens and settings
```

---

## Quick Start

### 1) Prepare Interest Group List

Input: `data/Interest_groups_manually_validated.xlsx` with columns:
`org_id`, `original_name_2`, `current_name_2`, `acronym_2`.

Run the prep script to:

* fill `current_name_2` where blank using `original_name_2`
* keep only `org_id`, `interest_group`, `acronym`
* output `data/interest_groups_list.csv`

**Windows PowerShell**

```powershell
Set-Location .\interest_group_analysis\1.data_collection
python .\interest_group_prep.py `
  --in  "..\..\data\Interest_groups_manually_validated.xlsx" `
  --out "..\..\data\interest_groups_list.csv"
```

### 2) Extract Mentions (strict: canonical + acronym only)

Input:

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

Outputs:

* `data/processed/mentions_114/mentions.jsonl`

  * each line includes `org_id`, `interest_group`, `variation`, `is_acronym`, `sentence`, etc.

---

## Reproducibility

* **Deterministic filters:** extractor uses whole-phrase matches only (no fuzzy/alt names), reducing false positives like generic “mental health.”
* **Versioned sample data:** tiny reproducible examples live in `data/sample/`.
* **Ignore large artifacts:** `.gitignore` excludes raw/processed data; retain provenance via scripts + configs.
* **Environment capture:** use `requirements.txt` (or pin versions in `pyproject.toml`) for consistent installs.

---

## License

Distributed under the terms of the **MIT License** (see `LICENSE`).

---

## Citation

If you use this toolkit in academic work, please cite this repository and the upstream APIs (GovInfo, Congress APIs, Google Trends).


