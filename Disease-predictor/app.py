import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from pages import diabetes, heart, alzheimer, report

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Disease Predictor")
    st.caption("ML-powered prediction system")
    st.divider()

    page = st.radio(
        "📌 Select Module",
        ["🩸 Diabetes", "❤️ Heart Disease", "🧠 Alzheimer's", "📊 Combined Report"],
        label_visibility="visible"
    )

    st.divider()
    st.caption("⚕️ For educational use only.\nNot a substitute for medical advice.")

# ── Route ─────────────────────────────────────────────────────────────────────
if page == "🩸 Diabetes":
    diabetes.show()
elif page == "❤️ Heart Disease":
    heart.show()
elif page == "🧠 Alzheimer's":
    alzheimer.show()
elif page == "📊 Combined Report":
    report.show()