"""Streamlit frontend for the committed, pre-contact prediction pipeline."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tourism_project.schema import CATEGORY_OPTIONS, PRE_CONTACT_FEATURES

DEPLOYMENT = Path(__file__).resolve().parent
MODEL_PATH = DEPLOYMENT / "model.joblib"
METADATA_PATH = DEPLOYMENT / "model_metadata.json"

st.set_page_config(page_title="Visit with Us", layout="centered")
st.title("Visit with Us: Package Purchase Prediction")
st.caption("Pre-contact decision support using existing customer-profile data.")
st.info(
    "Enter the customer's pre-contact profile details below to estimate "
    "their likelihood of purchasing the Wellness Tourism Package."
)
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Run the training pipeline before deploying.")
    st.stop()
metadata = json.loads(METADATA_PATH.read_text())
if metadata["model_scope"] != "pre_contact":
    st.error("This frontend requires the pre-contact model.")
    st.stop()
if metadata["packages"]["scikit-learn"] != version("scikit-learn"):
    st.error("Install the pinned deployment requirements to match model training.")
    st.stop()


@st.cache_resource
def load_model(path, modified_ns, fingerprint):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != fingerprint:
        raise ValueError("Model fingerprint does not match its metadata.")
    return joblib.load(path)


model = load_model(
    str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns, metadata["model_sha256"]
)
labels = {
    "Age": "Age",
    "CityTier": "City tier",
    "Occupation": "Occupation",
    "Gender": "Gender",
    "NumberOfPersonVisiting": "Total people visiting (including children)",
    "PreferredPropertyStar": "Preferred hotel stars",
    "MaritalStatus": "Marital status",
    "NumberOfTrips": "Trips per year",
    "Passport": "Valid passport",
    "OwnCar": "Owns a car",
    "NumberOfChildrenVisiting": "Children below age 5 visiting",
    "Designation": "Designation",
    "MonthlyIncome": "Monthly income (dataset currency units)",
}
numeric_widgets = {
    "Age": (18, 120, 35),
    "NumberOfPersonVisiting": (1, 20, 3),
    "NumberOfTrips": (0, 52, 3),
    "NumberOfChildrenVisiting": (0, 20, 1),
    "MonthlyIncome": (0, 1000000, 25000),
}
inputs = {}
with st.form("customer_profile"):
    for feature in PRE_CONTACT_FEATURES:
        if feature in CATEGORY_OPTIONS:
            inputs[feature] = st.selectbox(
                labels[feature], CATEGORY_OPTIONS[feature], key=feature
            )
        elif feature in ["Passport", "OwnCar"]:
            choice = st.selectbox(labels[feature], ["No", "Yes"], key=feature)
            inputs[feature] = int(choice == "Yes")
        elif feature in ["CityTier", "PreferredPropertyStar"]:
            options = [1, 2, 3] if feature == "CityTier" else [3, 4, 5]
            inputs[feature] = st.selectbox(labels[feature], options, key=feature)
        else:
            low, high, default = numeric_widgets[feature]
            inputs[feature] = st.number_input(
                labels[feature],
                min_value=low,
                max_value=high,
                value=default,
                step=1,
                key=feature,
            )
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    if inputs["NumberOfChildrenVisiting"] > inputs["NumberOfPersonVisiting"]:
        st.error("Children cannot exceed the total number of people visiting.")
    else:
        customer = pd.DataFrame([inputs], columns=metadata["feature_names"])
        outside_range = [
            labels[name]
            for name, (minimum, maximum) in metadata["numeric_training_ranges"].items()
            if not minimum <= inputs[name] <= maximum
        ]
        if outside_range:
            st.warning(
                "Outside the observed training range: "
                + ", ".join(outside_range)
                + ". Treat this prediction cautiously."
            )
        probability = float(model.predict_proba(customer)[0, 1])
        threshold = metadata["threshold"]
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        st.write(
            "Predicted purchase"
            if probability >= threshold
            else "Predicted no purchase"
        )
        st.caption(
            f"Fixed decision threshold: {threshold:.0%}. "
            "Random Forest probabilities are not separately calibrated."
        )
        st.dataframe(customer, hide_index=True)
st.caption(
    "Predict loads the committed model; it does not trigger training. "
    "Review performance, fairness, consent, and drift before operational use."
)
