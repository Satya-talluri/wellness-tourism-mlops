"""Clean categories and save a reproducible, profile-grouped 70/30 split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tourism_project.schema import (
    DATA_PATH,
    POST_PITCH_FEATURES,
    PRE_CONTACT_FEATURES,
    ROOT,
    TARGET,
    validate_data,
)


def prepare_data(data_path=DATA_PATH, output_dir=None):
    output_dir = Path(output_dir or ROOT / "artifacts" / "data-splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    cleaned = validate_data(raw)
    exact_duplicates = int(raw.duplicated().sum())
    cleaned = cleaned.loc[~raw.duplicated()].reset_index(drop=True)
    features = cleaned[POST_PITCH_FEATURES]
    target = cleaned[TARGET].astype(int)

    # Different customer IDs can describe identical profiles. Group them rather
    # than assuming that distinct customers are erroneous duplicate records.
    groups = pd.util.hash_pandas_object(
        cleaned[PRE_CONTACT_FEATURES], index=False
    ).astype(str)
    if groups.nunique() < 10:
        raise ValueError("At least 10 distinct customer profiles are needed.")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(splitter.split(features, target, groups))
    test_indices = np.sort(np.concatenate([test for _, test in folds[:3]]))
    train_indices = np.setdiff1d(np.arange(len(cleaned)), test_indices)

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    if train_groups & test_groups:
        raise RuntimeError("Customer profiles overlap between train and test.")
    for name, indices in [("train", train_indices), ("test", test_indices)]:
        if target.iloc[indices].nunique() != 2:
            raise ValueError(f"Both target classes are required in {name}.")
        features.iloc[indices].to_csv(output_dir / f"X{name}.csv", index=False)
        target.iloc[indices].to_csv(output_dir / f"y{name}.csv", index=False)
        groups.iloc[indices].rename("profile_group").to_csv(
            output_dir / f"groups_{name}.csv", index=False
        )
    summary = {
        "source_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "source_rows": len(raw),
        "exact_duplicates_removed": exact_duplicates,
        "repeated_rows_without_identifiers_before_cleaning": int(
            raw.drop(
                columns=["CustomerID"]
                + [column for column in raw if column.startswith("Unnamed:")]
            )
            .duplicated()
            .sum()
        ),
        "gender_aliases_corrected": int(raw["Gender"].eq("Fe Male").sum()),
        "marital_aliases_merged": int(raw["MaritalStatus"].eq("Unmarried").sum()),
        "cleaned_rows": len(cleaned),
        "distinct_pre_contact_profiles": int(groups.nunique()),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "test_fraction": len(test_indices) / len(cleaned),
        "train_purchase_rate": float(target.iloc[train_indices].mean()),
        "test_purchase_rate": float(target.iloc[test_indices].mean()),
        "overlapping_profile_groups": len(train_groups & test_groups),
        "pre_contact_features": PRE_CONTACT_FEATURES,
        "post_pitch_features": POST_PITCH_FEATURES,
        "split_method": "First 3 of 10 stratified group folds held out",
        "random_state": 42,
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.glob("*.csv"))
        },
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "data-splits"
    )
    args = parser.parse_args()
    prepare_data(args.data, args.output_dir)
