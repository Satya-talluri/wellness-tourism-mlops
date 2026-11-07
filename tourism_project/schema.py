"""Dataset contract and deterministic cleaning shared by the pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
TARGET = "ProdTaken"
EXPECTED_COLUMNS = [
    "CustomerID",
    TARGET,
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]
INTERACTION_COLUMNS = [
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]
CATEGORY_OPTIONS = {
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Female", "Male"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "ProductPitched": ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
}
POST_PITCH_FEATURES = [
    column for column in EXPECTED_COLUMNS if column not in ["CustomerID", TARGET]
]
PRE_CONTACT_FEATURES = [
    column for column in POST_PITCH_FEATURES if column not in INTERACTION_COLUMNS
]
NUMERIC_COLUMNS = [
    column for column in EXPECTED_COLUMNS if column not in CATEGORY_OPTIONS
]


def normalize_data(frame):
    cleaned = frame.copy()
    for column in CATEGORY_OPTIONS:
        if column in cleaned:
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    if "Gender" in cleaned:
        cleaned["Gender"] = cleaned["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in cleaned:
        cleaned["MaritalStatus"] = cleaned["MaritalStatus"].replace(
            {"Unmarried": "Single"}
        )
    return cleaned


def validate_data(frame):
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra = [
        column
        for column in frame.columns
        if column not in EXPECTED_COLUMNS and not column.startswith("Unnamed:")
    ]
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={missing}, unexpected={extra}")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only nonmissing 0 and 1 values.")
    if frame["CustomerID"].isna().any():
        raise ValueError("CustomerID must not be missing.")

    cleaned = normalize_data(frame)
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & values.isna()).any():
            raise ValueError(f"{column} contains nonnumeric values.")
        if np.isinf(values).any():
            raise ValueError(f"{column} contains infinite values.")
        if (values.dropna() < 0).any():
            raise ValueError(f"{column} contains negative values.")
    for column, choices in CATEGORY_OPTIONS.items():
        unknown = set(cleaned[column].dropna()) - set(choices)
        if unknown:
            raise ValueError(f"Unexpected {column} categories: {sorted(unknown)}")
    for column in ["Passport", "OwnCar"]:
        if not cleaned[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must be 0 or 1.")
    for column, choices in {
        "CityTier": [1, 2, 3],
        "PreferredPropertyStar": [3, 4, 5],
        "PitchSatisfactionScore": [1, 2, 3, 4, 5],
    }.items():
        if not cleaned[column].dropna().isin(choices).all():
            raise ValueError(f"Invalid {column} values.")
    if (cleaned["NumberOfChildrenVisiting"] > cleaned["NumberOfPersonVisiting"]).any():
        raise ValueError("Children cannot exceed the total visiting party.")
    return cleaned[EXPECTED_COLUMNS].copy()
