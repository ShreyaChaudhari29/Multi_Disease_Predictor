import os
import streamlit as st
import pandas as pd
from pages.diabetes   import load_model as diab_model
from pages.heart      import load_model as heart_model
from pages.alzheimer  import load_model as alz_model


# ── File Handling — Save report to disk ───────────────────────────────────────
def save_report(rows, path="combined_report.txt"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 40 + "\n")
            f.write("  MULTI DISEASE — COMBINED REPORT\n")
            f.write("=" * 40 + "\n\n")
            for r in rows:
                for key, val in r.items():
                    f.write(f"{key:<15}: {val}\n")
                f.write("-" * 40 + "\n\n")
        return path
    except Exception as e:
        st.error(f"Could not save report: {e}")
        return None


def show():
    st.title("📊 Combined Disease Report")
    st.caption("Auto-loads all 3 models and generates a unified report.")
    st.divider()

    if st.button("📋 Generate Combined Report", use_container_width=True):
        rows = []

        with st.spinner("Loading models and generating report..."):
            configs = [
                ("🩸 Diabetes",    diab_model,  "Random Forest"),
                ("❤️ Heart",       heart_model, "Logistic Regression"),
                ("🧠 Alzheimer's", alz_model,   "Decision Tree"),
            ]
            for name, loader, model_name in configs:
                try:
                    model, acc = loader()
                    if model and acc:
                        rows.append({
                            "Disease" : name,
                            "Model"   : model_name,
                            "Accuracy": f"{acc*100:.1f}%",
                        })
                except Exception as e:
                    st.warning(f"Could not load {name}: {e}")

        if not rows:
            st.error("Could not load any models.")
            return

        # ── Summary Table ─────────────────────────────────────────────────
        st.subheader("📋 Report Summary")
        st.dataframe(pd.DataFrame(rows).set_index("Disease"), use_container_width=True)

        # ── Metric Cards ──────────────────────────────────────────────────
        st.subheader("📈 Model Accuracy")
        mcols = st.columns(len(rows))
        for i, row in enumerate(rows):
            mcols[i].metric(row["Disease"], row["Accuracy"], row["Model"])

        # ── Save & Download ───────────────────────────────────────────────
        st.divider()
        path = save_report(rows)
        if path:
            st.success(f"✅ Report saved to `{path}`")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            st.download_button("⬇️ Download Report (.txt)", data=text,
                               file_name="combined_report.txt", mime="text/plain",
                               use_container_width=True)
            with st.expander("📄 View Report File"):
                st.code(text, language="text")