import json
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_catboost.cbm"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


model, metadata = load_model()
threshold = metadata["threshold"]


st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Classification")
st.caption("An educational CatBoost classification project")

st.warning(
    "This application is an educational demonstration, not a medical "
    "diagnostic tool. Its output must not replace professional advice."
)

with st.sidebar:
    st.header("Model information")
    st.metric("Repeated-CV accuracy", "84.8%", help="Mean accuracy across repeated validation folds.")
    st.metric("Repeated-CV ROC-AUC", "90.4%", help="Ability to rank positive and negative observations.")
    st.metric("Holdout recall", "93.9%", help="31 of 33 positive holdout observations were detected.")
    st.caption("Dataset: 303 observations · Holdout: 61 observations")

    with st.expander("How to read the result"):
        st.write(
            "The percentage is a model score for the entered values. "
            "It is not the model accuracy and is not a clinical risk estimate."
        )

st.subheader("Participant information")
st.write("Complete the fields below, then run the experimental prediction.")

with st.form("prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### General measurements")
        age = st.number_input("Age", min_value=18, max_value=100, value=54)

        sex_label = st.selectbox("Sex", ["Female", "Male"])
        sex = 0 if sex_label == "Female" else 1

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=70,
            max_value=250,
            value=130,
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=100,
            max_value=700,
            value=245,
        )

        fbs_label = st.selectbox(
            "Fasting blood sugar above 120 mg/dL",
            ["No", "Yes"],
        )
        fbs = 1 if fbs_label == "Yes" else 0

        thalach = st.number_input(
            "Maximum heart rate",
            min_value=60,
            max_value=220,
            value=150,
        )

    with right:
        st.markdown("#### Examination results")
        cp_labels = {
            "Typical angina (0)": 0,
            "Atypical angina (1)": 1,
            "Non-anginal pain (2)": 2,
            "Asymptomatic (3)": 3,
        }
        cp = cp_labels[st.selectbox("Chest pain type", list(cp_labels))]

        restecg_labels = {
            "Normal (0)": 0,
            "ST-T abnormality (1)": 1,
            "Left ventricular hypertrophy (2)": 2,
        }
        restecg = restecg_labels[
            st.selectbox("Resting ECG result", list(restecg_labels))
        ]

        exang_label = st.selectbox("Exercise-induced angina", ["No", "Yes"])
        exang = 1 if exang_label == "Yes" else 0

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        slope_labels = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
        slope = slope_labels[
            st.selectbox("Peak exercise ST slope", list(slope_labels))
        ]

        ca = st.selectbox("Number of major vessels", options=[0, 1, 2, 3, 4])

        thal_labels = {
            "Unknown/unspecified (0)": 0,
            "Fixed defect (1)": 1,
            "Normal (2)": 2,
            "Reversible defect (3)": 3,
        }
        thal = thal_labels[st.selectbox("Thal category", list(thal_labels))]

    submitted = st.form_submit_button(
        "Run experimental prediction",
        type="primary",
        use_container_width=True,
    )


if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= threshold)
    predicted_label = "Heart disease" if prediction == 1 else "No heart disease"
    predicted_confidence = probability if prediction == 1 else 1 - probability

    st.divider()
    st.subheader("Experimental model output")

    score_column, interpretation_column = st.columns([1, 2], gap="large")
    with score_column:
        st.metric(
            f"Predicted class: {predicted_label}",
            f"{predicted_confidence:.1%} confidence",
        )
        st.progress(
            float(predicted_confidence),
            text="Model confidence for the predicted class",
        )
        st.caption(f"Positive-class score: {probability:.1%}")

    with interpretation_column:
        distance_from_threshold = abs(probability - threshold)

        if distance_from_threshold < 0.10:
            st.warning(
                f"**{predicted_label} with {predicted_confidence:.1%} model "
                "confidence — uncertain.** The score is close to the 50% "
                "threshold, so small input changes could change the class."
            )
        elif prediction == 1:
            st.error(
                f"**Heart disease with {predicted_confidence:.1%} model "
                "confidence.**"
            )
        else:
            st.success(
                f"**No heart disease with {predicted_confidence:.1%} model "
                "confidence.**"
            )

        st.write(f"**Distance from threshold:** {distance_from_threshold:.1%}")

    if probability >= threshold:
        class_explanation = "the score is equal to or above the threshold"
    else:
        class_explanation = "the score is below the threshold"

    st.caption(
        f"Threshold: {threshold:.0%}; {class_explanation}. "
        "This score is not a diagnosis or an estimated clinical risk."
    )

with st.expander("About this project"):
    st.write(
        "The CatBoost model was trained on 303 observations and evaluated with "
        "repeated stratified cross-validation. Its mean validation accuracy was "
        "84.8%, with substantial uncertainty due to the small dataset."
    )
    st.write(
        "A score such as 58.8% does not mean the model is 58.8% accurate. "
        "It means this particular example lies only 8.8 percentage points above "
        "the model's decision threshold."
    )
