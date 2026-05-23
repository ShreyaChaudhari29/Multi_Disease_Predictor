import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ── File Handling ─────────────────────────────────────────────────────────────
def load_csv(label, filename):
    """Upload CSV via Streamlit file uploader."""
    f = st.file_uploader(f"📂 Upload `{filename}`", type="csv", key=label)
    if f:
        return pd.read_csv(f)
    return None


# ── ML Helpers ────────────────────────────────────────────────────────────────
def split_features_target(df, target_col):
    return df.drop(target_col, axis=1), df[target_col]


def train_and_evaluate(model, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc


# ── Pure Python UI Components ─────────────────────────────────────────────────
def show_accuracy(model_name, acc):
    """Show model accuracy using Streamlit metric."""
    col1, col2 = st.columns(2)
    col1.metric("🤖 Model", model_name)
    col2.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    st.divider()


def show_inputs(X):
    """Render dynamic number inputs in 2 columns using pure Streamlit."""
    inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            inputs[col] = st.number_input(
                label=col,
                value=float(X[col].mean()),
                format="%.2f"
            )
    return inputs


def show_result(pred, positive_msg, negative_msg, positive_val=1):
    """Show prediction result using Streamlit success/error."""
    st.divider()
    if pred == positive_val:
        st.error(f"⚠️  {positive_msg}")
    else:
        st.success(f"✅  {negative_msg}")