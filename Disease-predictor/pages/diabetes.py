import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io, requests

# ── Load & Train once (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("diabetes (1).csv")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc


def show():
    st.title("🩸 Diabetes Prediction")
    st.caption("Model: Random Forest  |  Dataset: PIMA Indians Diabetes (auto-loaded)")
    st.divider()

    model, acc = load_model()
    if model is None:
        return

    col1, col2 = st.columns(2)
    col1.metric("🤖 Model", "Random Forest")
    col2.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    st.divider()

    # ── Input Fields ──────────────────────────────────────────────────────────
    st.subheader("📋 Enter Patient Details")

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies",           min_value=0,   max_value=20,   value=1)
        glucose     = st.number_input("Glucose Level",         min_value=0,   max_value=300,  value=110)
        bp          = st.number_input("Blood Pressure",        min_value=0,   max_value=200,  value=70)
        skin        = st.number_input("Skin Thickness",        min_value=0,   max_value=100,  value=20)
    with c2:
        insulin     = st.number_input("Insulin Level",         min_value=0,   max_value=900,  value=80)
        bmi         = st.number_input("BMI",                   min_value=0.0, max_value=70.0, value=25.0)
        dpf         = st.number_input("Diabetes Pedigree",     min_value=0.0, max_value=3.0,  value=0.5)
        age         = st.number_input("Age",                   min_value=1,   max_value=120,  value=30)

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Predict Diabetes", use_container_width=True):
        data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                            columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                     "Insulin","BMI","DiabetesPedigreeFunction","Age"])
        pred = model.predict(data)[0]
        # Probability
        probabilities = model.predict_proba(data)[0]

        confidence = max(probabilities) * 100
        # Show confidence
        st.info(f"Prediction Confidence: {confidence:.2f}%")
        st.divider()
        if pred == 1:
            st.error(
                f"⚠️ Diabetic — Please consult a doctor. "
                f"({confidence:.1f}% probability)"
            )
        else:
            st.success(
                f"✅ Not Diabetic — You're doing great! "
                f"({confidence:.1f}% confidence)"
            )
        # Detailed probabilities
        st.write(
            f"Healthy Probability: "
            f"{probabilities[0]*100:.2f}%"
        )
        st.write(
            f"Diabetes Probability: "
            f"{probabilities[1]*100:.2f}%"
        )