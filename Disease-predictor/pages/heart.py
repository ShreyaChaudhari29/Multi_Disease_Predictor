import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Load & Train once (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("heart_disease.csv")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None

    target = "target" if "target" in df.columns else df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc


def show():
    st.title("❤️ Heart Disease Prediction")
    st.caption("Model: Logistic Regression  |  Dataset: Cleveland Heart Disease (auto-loaded)")
    st.divider()

    model, acc = load_model()
    if model is None:
        return

    col1, col2 = st.columns(2)
    col1.metric("🤖 Model", "Logistic Regression")
    col2.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    st.divider()

    # ── Input Fields ──────────────────────────────────────────────────────────
    st.subheader("📋 Enter Patient Details")

    c1, c2 = st.columns(2)
    with c1:
        age     = st.number_input("Age",                    min_value=1,   max_value=120, value=45)
        sex     = st.selectbox("Sex",                       options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp      = st.selectbox("Chest Pain Type",           options=[0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
        trestbps= st.number_input("Resting Blood Pressure", min_value=80,  max_value=250, value=120)
        chol    = st.number_input("Cholesterol (mg/dl)",    min_value=100, max_value=600, value=200)
        fbs     = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG",               options=[0,1,2])
    with c2:
        thalach = st.number_input("Max Heart Rate",         min_value=60,  max_value=250, value=150)
        exang   = st.selectbox("Exercise Induced Angina",   options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression",          min_value=0.0, max_value=10.0,value=1.0)
        slope   = st.selectbox("Slope of ST Segment",       options=[0,1,2])
        ca      = st.selectbox("Major Vessels (0-3)",       options=[0,1,2,3])
        thal    = st.selectbox("Thalassemia",               options=[0,1,2,3], format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","?"][x])

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]],
                            columns=["age","sex","cp","trestbps","chol","fbs",
                                     "restecg","thalach","exang","oldpeak","slope","ca","thal"])
        pred = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence = max(probabilities) * 100

        st.info(f"Prediction Confidence: {confidence:.2f}%")
        st.divider()
        if pred == 1:
            st.error(
                f"⚠️ Heart Disease Detected — "
                f"Please seek medical advice. "
                f"({confidence:.1f}% probability)"
            )

        else:
            st.success(
                f"✅ No Heart Disease — "
                f"Heart looks healthy! "
                f"({confidence:.1f}% confidence)"
            )
        # Detailed probabilities
        st.write(
            f"Healthy Probability: "
            f"{probabilities[0]*100:.2f}%"
        )

        st.write(
            f"Heart Disease Probability: "
            f"{probabilities[1]*100:.2f}%"
        )