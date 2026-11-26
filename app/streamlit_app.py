"""Streamlit app: enter a patient's clinical data, get a heart disease risk estimate.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import FEATURES  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

HIGH_RISK_CUTOFF = 0.70

# Display label -> value the model was trained on
CHEST_PAIN = {
    "Typical angina": "typical angina",
    "Atypical angina": "atypical angina",
    "Non-anginal pain": "non-anginal",
    "Asymptomatic (no chest pain)": "asymptomatic",
}
RESTING_ECG = {
    "Normal": "normal",
    "ST-T wave abnormality": "st-t abnormality",
    "Left ventricular hypertrophy": "lv hypertrophy",
}
YES_NO_UNKNOWN = {"No": "no", "Yes": "yes", "Unknown": np.nan}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalch": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
}
UNKNOWN_HELP = "If unknown, the model assumes the most common answer in the training data."


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def risk_level(proba, threshold):
    if proba >= HIGH_RISK_CUTOFF:
        return "High", "error"
    if proba >= threshold:
        return "Moderate", "warning"
    return "Low", "success"


def explain(model, patient):
    """Per-feature contribution to the prediction (log-odds), for linear models.

    Each contribution = coefficient x the patient's preprocessed value, i.e. how
    far that feature moves the risk compared with an average patient.
    Returns None for models without coefficients (e.g. tree ensembles).
    """
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None

    prep = model.named_steps["prep"]
    values = prep.transform(patient)[0]
    contributions = values * clf.coef_[0]

    # Map one-hot columns ("cat__cp_asymptomatic") back to their feature ("cp")
    totals = {}
    for name, c in zip(prep.get_feature_names_out(), contributions):
        feature = next(f for f in FEATURES if name.split("__", 1)[1].startswith(f))
        totals[feature] = totals.get(feature, 0.0) + c
    return pd.Series(totals).sort_values(key=abs, ascending=False)


DISPLAY_VALUES = {
    "cp": {v: k for k, v in CHEST_PAIN.items()},
    "restecg": {v: k for k, v in RESTING_ECG.items()},
    "fbs": {"yes": "Yes", "no": "No"},
    "exang": {"yes": "Yes", "no": "No"},
}


def format_value(feature, patient):
    value = patient[feature].iloc[0]
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        return f"{value:g}"
    return DISPLAY_VALUES.get(feature, {}).get(value, str(value))


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.train` and `python -m src.evaluate` first.")
    st.stop()

model, metadata = load_model()
threshold = metadata["threshold"]

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {metadata['model_name']}")
    test = metadata.get("test_metrics", {})
    if test:
        st.write(f"**Test ROC-AUC:** {test['roc_auc']:.2f}")
        st.write(f"**Test recall:** {test['recall']:.0%}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    st.caption(
        "Trained on the UCI Heart Disease dataset (920 patients, 4 hospitals). "
        "The threshold was lowered from 0.5 to favour catching sick patients over avoiding false alarms."
    )

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter the patient's clinical information to estimate their risk of heart disease.")

with st.form("patient_form"):
    st.subheader("Patient information")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=54)
    sex = c2.selectbox("Sex", ["Male", "Female"])

    st.subheader("Symptoms")
    c1, c2 = st.columns(2)
    cp = c1.selectbox("Chest pain type", list(CHEST_PAIN), index=3)
    exang = c2.selectbox("Chest pain during exercise (angina)?", list(YES_NO_UNKNOWN), help=UNKNOWN_HELP)

    st.subheader("Measurements")
    c1, c2 = st.columns(2)
    trestbps = c1.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=220, value=130)
    thalch = c2.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=140)
    chol = c1.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=610, value=230)
    chol_unknown = c1.checkbox("Cholesterol not known", help="The model will use the median cholesterol instead.")
    fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl?", list(YES_NO_UNKNOWN), help=UNKNOWN_HELP)

    st.subheader("ECG / exercise test")
    c1, c2 = st.columns(2)
    restecg = c1.selectbox("Resting ECG result", list(RESTING_ECG))
    oldpeak = c2.number_input(
        "ST depression induced by exercise (oldpeak)",
        min_value=-3.0, max_value=7.0, value=0.5, step=0.1,
        help="ST depression relative to rest, from an exercise ECG. 0 means none.",
    )

    submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

if submitted:
    patient = pd.DataFrame([{
        "age": age,
        "trestbps": float(trestbps),
        "chol": np.nan if chol_unknown else float(chol),
        "thalch": float(thalch),
        "oldpeak": float(oldpeak),
        "sex": sex,
        "cp": CHEST_PAIN[cp],
        "fbs": YES_NO_UNKNOWN[fbs],
        "restecg": RESTING_ECG[restecg],
        "exang": YES_NO_UNKNOWN[exang],
    }])[FEATURES]

    proba = float(model.predict_proba(patient)[0, 1])
    level, style = risk_level(proba, threshold)

    st.divider()
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Estimated risk probability", f"{proba:.0%}")
    c2.metric("Risk level", level)
    st.progress(proba)
    getattr(st, style)(
        {
            "High": "High estimated risk of heart disease. A cardiology consultation is strongly recommended.",
            "Moderate": "Moderate estimated risk. Above the screening threshold — further medical tests are advisable.",
            "Low": "Low estimated risk. Below the screening threshold.",
        }[level]
    )

    contributions = explain(model, patient)
    if contributions is not None:
        st.subheader("What drove this prediction")
        up = contributions[contributions > 0].head(3)
        down = contributions[contributions < 0].head(3)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**⬆️ Increasing risk**")
            for f in up.index:
                st.write(f"- {FEATURE_LABELS[f]}: {format_value(f, patient)}")
            if up.empty:
                st.write("- None")
        with c2:
            st.markdown("**⬇️ Decreasing risk**")
            for f in down.index:
                st.write(f"- {FEATURE_LABELS[f]}: {format_value(f, patient)}")
            if down.empty:
                st.write("- None")
        st.caption("Based on the model's learned weights applied to this patient's values.")

    with st.expander("How are risk levels defined?"):
        st.write(
            f"- **Low:** probability below {threshold:.0%} (the screening threshold)\n"
            f"- **Moderate:** {threshold:.0%} – {HIGH_RISK_CUTOFF:.0%}\n"
            f"- **High:** {HIGH_RISK_CUTOFF:.0%} or above\n\n"
            f"The threshold was chosen so the model catches about 90% of patients with heart disease."
        )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is an educational machine learning project, not a medical device. "
    "It must not be used for diagnosis or treatment decisions. Always consult a qualified healthcare professional."
)
