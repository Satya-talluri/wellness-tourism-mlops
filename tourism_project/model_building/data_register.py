"""Validate the untouched source CSV and record its fingerprint."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tourism_project.schema import DATA_PATH, ROOT, TARGET, validate_data


def register_data(data_path=DATA_PATH, report_path=None):
    data_path = Path(data_path)
    raw = pd.read_csv(data_path)
    validate_data(raw)
    summary = {
        "dataset": data_path.name,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows": len(raw),
        "raw_columns": len(raw.columns),
        "columns": raw.columns.tolist(),
        "missing_values": raw.isna().sum().to_dict(),
        "target_counts": raw[TARGET].value_counts().sort_index().to_dict(),
        "purchase_rate": float(raw[TARGET].mean()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw["CustomerID"].duplicated().sum()),
        "extra_index_columns": [
            column for column in raw.columns if column.startswith("Unnamed:")
        ],
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "registration.json"
    )
    args = parser.parse_args()
    register_data(args.data, args.report)
