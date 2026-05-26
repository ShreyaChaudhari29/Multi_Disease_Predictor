import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Load & Train once (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        # FIX 1: na_values=["N/A"] so "N/A" strings become proper NaN
        df = pd.read_csv("alzheimer.csv", na_values=["N/A"])
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None, None

    # FIX 2: Your CSV has no "Group" column — CDR is the correct target.
    # CDR = Clinical Dementia Rating: 0.0=None, 0.5=Very Mild, 1.0=Mild, 2.0=Moderate
    target = "CDR"

    # FIX 3: Drop rows where the TARGET is NaN (201 rows had CDR=NaN)
    df = df.dropna(subset=[target])

    # FIX 4: Convert CDR to labelled strings so LabelEncoder gives meaningful output
    cdr_map = {0.0: "No Dementia", 0.5: "Very Mild", 1.0: "Mild", 2.0: "Moderate"}
    df[target] = df[target].map(cdr_map)

    # Drop columns not useful for prediction
    drop_cols = ["ID", "Hand", "Delay"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Fill remaining missing numeric values with column median
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()  # drop any leftover rows

    # Encode M/F column
    if "M/F" in df.columns:
        df["M/F"] = LabelEncoder().fit_transform(df["M/F"])

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target])

    # Features — numeric only
    X = df.drop(target, axis=1).select_dtypes(include=["number"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc, target_encoder


def show():
    st.title("🧠 Alzheimer's Prediction")
    st.caption("Model: Decision Tree  |  Dataset: OASIS Alzheimer's (auto-loaded)")
    st.divider()

    model, acc, target_encoder = load_model()
    if model is None:
        return

    col1, col2 = st.columns(2)
    col1.metric("🤖 Model", "Decision Tree")
    col2.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    st.divider()

    # ── Input Fields ──────────────────────────────────────────────────────────
    st.subheader("📋 Enter Patient Details")

    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age",                       min_value=18,  max_value=100, value=70)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_value = 1 if gender == "Male" else 0
        educ   = st.selectbox("Education Level (years)",      options=[1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}")
        ses    = st.selectbox("Socioeconomic Status",         options=[1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}")
        mmse   = st.number_input("MMSE Score (0-30)",         min_value=0,   max_value=30,  value=25)
    with c2:
        etiv   = st.number_input("Estimated Total Intracranial Volume", min_value=1000, max_value=2000, value=1450)
        nwbv   = st.number_input("Normalized Whole Brain Volume",       min_value=0.5,  max_value=1.0,  value=0.75, format="%.3f")
        asf    = st.number_input("Atlas Scaling Factor",                min_value=0.8,  max_value=1.6,  value=1.2,  format="%.3f")

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Predict Alzheimer's", use_container_width=True):
        # FIX 5: Input columns must match what the model was trained on (no CDR input field)
        data = pd.DataFrame(
            [[age, gender_value, educ, ses, mmse, etiv, nwbv, asf]],
            columns=["Age", "M/F", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
        )
        data = data[model.feature_names_in_]

        pred          = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence    = max(probabilities) * 100
        result        = target_encoder.inverse_transform([pred])[0]

        st.info(f"Prediction Confidence: {confidence:.2f}%")

        classes = target_encoder.classes_
        for cls, prob in zip(classes, probabilities):
            st.write(f"{cls}: {prob*100:.2f}%")

        st.divider()
        if result == "No Dementia":
            st.success(f"✅  {result} — No signs of Alzheimer's.")
        else:
            st.error(f"⚠️  {result} — Please consult a neurologist.")
