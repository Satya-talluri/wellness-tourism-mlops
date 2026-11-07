import json

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tourism_project.model_building.train import load_splits
from tourism_project.schema import PRE_CONTACT_FEATURES, ROOT

DEPLOYMENT = ROOT / "tourism_project" / "deployment"


@pytest.fixture
def trained_model():
    path = DEPLOYMENT / "model.joblib"
    if not path.exists():
        pytest.fail("Train the model before running deployment tests.")
    return joblib.load(path)


def test_saved_pipeline_handles_an_unseen_category(trained_model):
    customer = pd.read_csv(ROOT / "artifacts/data-splits/Xtest.csv").head(1)
    customer = customer[PRE_CONTACT_FEATURES].copy()
    customer.loc[:, "Occupation"] = "New occupation"
    probabilities = trained_model.predict_proba(customer)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_saved_pipeline_handles_missing_profile_values(trained_model):
    customer = pd.read_csv(ROOT / "artifacts/data-splits/Xtest.csv").head(1)
    customer = customer[PRE_CONTACT_FEATURES].copy()
    customer.loc[:, "Age"] = np.nan
    customer.loc[:, "Gender"] = np.nan
    assert np.isfinite(trained_model.predict_proba(customer)).all()


def test_metadata_matches_the_saved_inference_pipeline(trained_model):
    metadata = json.loads((DEPLOYMENT / "model_metadata.json").read_text())
    assert metadata["model_scope"] == "pre_contact"
    assert metadata["feature_names"] == trained_model.feature_names_in_.tolist()


def test_changed_artifact_is_rejected(tmp_path):
    source = ROOT / "artifacts/data-splits"
    for path in source.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    path = tmp_path / "Xtrain.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_splits(tmp_path)


def test_streamlit_predicts_using_the_committed_model():
    app = AppTest.from_file(str(DEPLOYMENT / "app.py")).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 1
    probability = float(app.metric[0].value.rstrip("%")) / 100
    assert 0 <= probability <= 1
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.columns.tolist() == PRE_CONTACT_FEATURES


def test_streamlit_rejects_impossible_visiting_party():
    app = AppTest.from_file(str(DEPLOYMENT / "app.py")).run(timeout=30)
    app.number_input(key="NumberOfPersonVisiting").set_value(1)
    app.number_input(key="NumberOfChildrenVisiting").set_value(2)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any("Children cannot exceed" in error.value for error in app.error)
    assert not app.metric
