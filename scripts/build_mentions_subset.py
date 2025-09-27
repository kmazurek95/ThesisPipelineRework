import json
from pathlib import Path

SRC = Path(r"c:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\data\processed\mentions_114\mentions.jsonl")
OUT = Path(r"c:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\data\processed\mentions_114\mentions_subset.jsonl")
WANT = {
    "CREC-2015-01-07-pt1-PgH56",
    "CREC-2015-01-07-pt1-PgH55-8",
    "CREC-2015-01-07-pt1-PgH59",
}

def main():
    count = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with SRC.open("r", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("granuleId") in WANT:
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} rows to {OUT}")

if __name__ == "__main__":
    main()
