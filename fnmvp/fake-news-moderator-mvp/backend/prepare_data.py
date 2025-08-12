import csv
import os
import random
from pathlib import Path

# Prepare train/test CSVs from Fake.csv and True.csv without extra dependencies
# Input files must exist at backend/data/Fake.csv and backend/data/True.csv
# Output files are backend/data/train.csv and backend/data/test.csv

BASE = Path(__file__).parent / "data"
FAKE = BASE / "Fake.csv"
TRUE = BASE / "True.csv"
TRAIN = BASE / "train.csv"
TEST = BASE / "test.csv"
IMAGES = BASE / "images"

SCHEMA_FIELDS = ["text", "label", "image_filename", "context"]


def read_rows(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    # Use utf-8-sig to handle BOM if present
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def normalize(rows, label_value: int):
    """Map source rows into unified schema. Picks 'text' column if present,
    otherwise tries known alternatives; falls back to empty string.
    """
    out = []
    for r in rows:
        text = (
            r.get("text")
            or r.get("content")
            or r.get("body")
            or r.get("article")
            or r.get("story")
            or ""
        )
        out.append({
            "text": text,
            "label": label_value,
            "image_filename": "",
            "context": "",
        })
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print(f"Reading {FAKE} and {TRUE} ...")
    fake_rows = normalize(read_rows(FAKE), 1)
    true_rows = normalize(read_rows(TRUE), 0)
    all_rows = fake_rows + true_rows

    if not all_rows:
        raise SystemExit("No rows found in inputs")

    # Shuffle deterministically
    random.Random(42).shuffle(all_rows)

    # Split 80/20 (ensure at least 1 test row if there are >=2 rows)
    split = max(1, int(0.8 * len(all_rows))) if len(all_rows) > 1 else 1
    train_rows = all_rows[:split]
    test_rows = all_rows[split:]

    # Ensure images directory exists (even if empty)
    IMAGES.mkdir(parents=True, exist_ok=True)

    write_csv(TRAIN, train_rows)
    write_csv(TEST, test_rows)

    print(f"Wrote {TRAIN} ({len(train_rows)} rows)")
    print(f"Wrote {TEST} ({len(test_rows)} rows)")
    print(f"Images folder: {IMAGES} (exists={IMAGES.exists()})")


if __name__ == "__main__":
    main()

