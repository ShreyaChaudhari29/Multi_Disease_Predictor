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
         df = pd.read_csv("alzheimer.csv")
         # Remove extra spaces from column names
         df.columns = df.columns.str.strip()
        
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None,None
    target = "Group"
    
    # Fill missing numeric values
    df = df.fillna(df.mean(numeric_only=True))
    # Remove remaining missing rows
    df = df.dropna()

    # Remove ID/text columns
    drop_cols = ["Subject ID", "MRI ID", "Visit", "Hand"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    # Encode only categorical input columns
    if "M/F" in df.columns:
        df["M/F"] = LabelEncoder().fit_transform(df["M/F"])
    # Encode target separately
    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(df[target])

    # Features
    X = df.drop(target, axis=1)

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])
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
        age   = st.number_input("Age",                      min_value=18,  max_value=100, value=70)
        
        gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

        gender_value = 1 if gender == "Male" else 0
        educ  = st.selectbox("Education Level (years)",     options=[1,2,3,4,5], format_func=lambda x: f"Level {x}")
        ses   = st.selectbox("Socioeconomic Status",        options=[1,2,3,4,5], format_func=lambda x: f"Level {x}")
        mmse  = st.number_input("MMSE Score (0-30)",        min_value=0,   max_value=30,  value=25)
    with c2:
        cdr   = st.selectbox("CDR (Dementia Rating)",       options=[0.0, 0.5, 1.0, 2.0], format_func=lambda x: f"{x} - {'Normal' if x==0 else 'Mild' if x==0.5 else 'Moderate' if x==1 else 'Severe'}")
        etiv  = st.number_input("Estimated Total Intracranial Volume", min_value=1000, max_value=2000, value=1450)
        nwbv  = st.number_input("Normalized Whole Brain Volume", min_value=0.5, max_value=1.0, value=0.75, format="%.3f")
        asf   = st.number_input("Atlas Scaling Factor",    min_value=0.8, max_value=1.6,  value=1.2, format="%.3f")

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Predict Alzheimer's", use_container_width=True):
        data = pd.DataFrame([[age, gender_value, educ, ses, mmse, cdr, etiv, nwbv, asf]],
                            columns=["Age","M/F","Educ","SES","MMSE","CDR","eTIV","nWBV","ASF"])
        data = data[model.feature_names_in_]

        pred = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence = max(probabilities) * 100
        result = target_encoder.inverse_transform([pred])[0]
        # Main confidence
        st.info(f"Prediction Confidence: {confidence:.2f}%")
        # Show all probabilities
        classes = target_encoder.classes_

        for cls, prob in zip(classes, probabilities):

            st.write(f"{cls}: {prob*100:.2f}%")

        st.divider()
        if "Non" in result:
            st.success(f"✅  {result} — No signs of Alzheimer's.")
        else:
            st.error(f"⚠️  {result} — Please consult a neurologist.")