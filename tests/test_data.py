import pandas as pd
import pytest

from tourism_project.model_building.prep import prepare_data
from tourism_project.schema import (
    DATA_PATH,
    INTERACTION_COLUMNS,
    PRE_CONTACT_FEATURES,
    normalize_data,
    validate_data,
)


@pytest.fixture
def sample():
    return pd.read_csv(DATA_PATH).head(3)


def test_alias_cleaning_does_not_modify_source(sample):
    sample.loc[0, "Gender"] = " Fe Male "
    sample.loc[0, "MaritalStatus"] = "Unmarried"
    cleaned = normalize_data(sample)
    assert cleaned.loc[0, "Gender"] == "Female"
    assert cleaned.loc[0, "MaritalStatus"] == "Single"
    assert sample.loc[0, "Gender"] == " Fe Male "
    assert sample.loc[0, "MaritalStatus"] == "Unmarried"


def test_registration_rejects_missing_target_column(sample):
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_data(sample.drop(columns="ProdTaken"))


@pytest.mark.parametrize("bad_target", [2, None])
def test_registration_rejects_invalid_targets(sample, bad_target):
    sample.loc[0, "ProdTaken"] = bad_target
    with pytest.raises(ValueError, match="ProdTaken"):
        validate_data(sample)


def test_registration_rejects_invalid_numeric_values(sample):
    sample["Age"] = sample["Age"].astype(object)
    sample.loc[0, "Age"] = "not a number"
    with pytest.raises(ValueError, match="nonnumeric"):
        validate_data(sample)


def test_registration_rejects_unrecognized_categories(sample):
    sample.loc[0, "Gender"] = "unexpected category"
    with pytest.raises(ValueError, match="Unexpected Gender"):
        validate_data(sample)


def test_grouped_preparation_prevents_profile_leakage(tmp_path):
    summary = prepare_data(output_dir=tmp_path)
    train = pd.read_csv(tmp_path / "Xtrain.csv")
    test = pd.read_csv(tmp_path / "Xtest.csv")
    train_profiles = set(
        pd.util.hash_pandas_object(train[PRE_CONTACT_FEATURES], index=False)
    )
    test_profiles = set(
        pd.util.hash_pandas_object(test[PRE_CONTACT_FEATURES], index=False)
    )
    assert train_profiles.isdisjoint(test_profiles)
    assert summary["overlapping_profile_groups"] == 0
    assert 0.27 < summary["test_fraction"] < 0.33
    assert "CustomerID" not in train
    assert not any(column.startswith("Unnamed:") for column in train)
    assert not set(PRE_CONTACT_FEATURES) & set(INTERACTION_COLUMNS)
    assert not train["Gender"].eq("Fe Male").any()
    assert not train["MaritalStatus"].eq("Unmarried").any()
    assert len(train) == len(pd.read_csv(tmp_path / "ytrain.csv"))
    assert len(test) == len(pd.read_csv(tmp_path / "ytest.csv"))
