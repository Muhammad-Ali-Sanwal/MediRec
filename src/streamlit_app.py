"""
Medicine Recommendation System – Streamlit Web App
Run: streamlit run src/streamlit_app.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.recommender import MedicineRecommender

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRec – Medicine Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .rec-card {
        background: #f8f9fa; border-left: 5px solid #2980b9;
        padding: 1rem 1.2rem; border-radius: 8px;
        margin-bottom: 0.8rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .severity-mild     { color: #27ae60; font-weight: bold; }
    .severity-moderate { color: #f39c12; font-weight: bold; }
    .severity-serious  { color: #e74c3c; font-weight: bold; }
    .disclaimer-box {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 1rem; margin-top: 1.5rem;
        font-size: 0.9rem; color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Recommender ─────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_recommender():
    return MedicineRecommender(
        model_dir=os.path.join(BASE, "models"),
        metadata_path=os.path.join(BASE, "data", "metadata.json")
    )

rec = load_recommender()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=80)
    st.title("MediRec")
    st.caption("AI-Powered Medicine Recommendation System")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Home & Recommend",
        "📊 Analytics Dashboard",
        "💊 Medicine Database",
        "ℹ️ About"
    ])
    st.divider()
    st.info("""**Supervisor:** Dr. Mushtaq Hussain
CS619 – Spring 2026

**Developer:** M. Ali Sanwal (DevOps Engineer)
**VUID:** BC240440384
**Portfolio:** [Link](https://sanwal.vercel.app/)""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Home & Recommend
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Recommend":
    st.markdown("""
    <div class="main-header">
        <h1>💊 Medicine Recommendation System</h1>
        <p>AI-powered personalized medicine suggestions based on your symptoms</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("Patient Information")

        with st.form("recommend_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 1, 100, 30)
            with c2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            severity = st.select_slider(
                "Symptom Severity",
                options=["Mild (1)", "Moderate (2)", "Serious (3)"],
                value="Moderate (2)"
            )
            severity_val = {"Mild (1)": 1, "Moderate (2)": 2, "Serious (3)": 3}[severity]

            symptoms_selected = st.multiselect(
                "Select Symptoms",
                options=rec.get_all_symptoms(),
                help="Select all symptoms the patient is experiencing"
            )

            history = st.text_area("Medical History (optional)",
                                   placeholder="e.g. Diabetic, hypertensive…", height=80)

            submitted = st.form_submit_button("🔍 Get Recommendations",
                                              use_container_width=True,
                                              type="primary")

    with col_result:
        if submitted:
            if not symptoms_selected:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Analyzing symptoms…"):
                    result = rec.recommend(
                        symptoms_selected, age, gender, severity_val, history
                    )

                st.subheader("🎯 Diagnosis & Recommendations")

                # Predicted Diseases
                diseases = result.get("predicted_diseases", [])
                if diseases:
                    st.markdown("**Predicted Conditions**")
                    for d in diseases[:3]:
                        pct = d["confidence"]
                        st.markdown(f"- **{d['disease']}** — {pct:.1f}% confidence")
                        st.progress(min(pct / 100, 1.0))

                st.divider()

                # Medicine Cards
                medicines = result.get("recommended_medicines", [])
                if medicines:
                    st.markdown("**Recommended Medicines**")
                    for i, med in enumerate(medicines, 1):
                        info = med.get("info", {})
                        with st.expander(
                            f"#{i} {med['medicine']} · {med.get('source','')}"
                        ):
                            if info:
                                st.markdown(f"**Class:** {info.get('drug_class','–')}")
                                st.markdown(f"**Dosage:** {info.get('dosage','–')}")
                                if info.get("side_effects"):
                                    st.markdown("**Side Effects:** " +
                                                ", ".join(info["side_effects"]))
                                if info.get("contraindications"):
                                    st.markdown("**⚠️ Contraindications:** " +
                                                ", ".join(info["contraindications"]))
                                st.markdown(f"**Category:** {info.get('category','–')}")
                            else:
                                st.info("Detailed info not available.")

                # Severity Badge
                sev = result.get("disease_severity", "")
                color = {"mild": "green", "moderate": "orange",
                         "serious": "red"}.get(sev, "gray")
                st.markdown(f"**Severity:** :{color}[{sev.upper()}]")

                # Confidence Chart
                if medicines:
                    conf_data = [m for m in medicines if m["confidence"] > 0]
                    if conf_data:
                        fig = go.Figure(go.Bar(
                            x=[m["medicine"] for m in conf_data],
                            y=[m["confidence"] for m in conf_data],
                            marker_color="#2980b9"
                        ))
                        fig.update_layout(
                            title="Medicine Confidence Scores",
                            xaxis_tickangle=-30, height=300,
                            margin=dict(l=20, r=20, t=40, b=60)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class="disclaimer-box">
                ⚠️ {result.get('disclaimer','')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Fill in the form on the left and click **Get Recommendations**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    # Load training results
    results_path = os.path.join(BASE, "models", "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            tr = json.load(f)

        st.success(f"Best Disease Model: **{tr['disease_best_model']}** | "
                   f"Best Medicine Model: **{tr['medicine_best_model']}**")

        # Model Comparison – Disease
        st.subheader("Disease Prediction – Model Comparison")
        disease_df = pd.DataFrame(tr["disease_results"]).T.reset_index()
        disease_df.columns = ["Model", "CV Mean", "CV Std", "Test Accuracy", "F1 Score"]
        fig1 = px.bar(disease_df, x="Model", y="Test Accuracy",
                      color="Test Accuracy", color_continuous_scale="Blues",
                      title="Test Accuracy by Model (Disease)")
        st.plotly_chart(fig1, use_container_width=True)

        # Model Comparison – Medicine
        st.subheader("Medicine Prediction – Model Comparison")
        med_df = pd.DataFrame(tr["medicine_results"]).T.reset_index()
        med_df.columns = ["Model", "CV Mean", "CV Std", "Test Accuracy", "F1 Score"]
        fig2 = px.bar(med_df, x="Model", y="F1 Score",
                      color="F1 Score", color_continuous_scale="Greens",
                      title="F1 Score by Model (Medicine)")
        st.plotly_chart(fig2, use_container_width=True)

        # DL Model
        if tr.get("deep_learning_accuracy"):
            st.metric("Deep Learning Model Accuracy",
                      f"{tr['deep_learning_accuracy']*100:.2f}%")
    else:
        st.warning("No training results found. Run `python models/train_models.py` first.")

    # Dataset stats
    data_path = os.path.join(BASE, "data", "patient_records.csv")
    if os.path.exists(data_path):
        st.subheader("Dataset Overview")
        df = pd.read_csv(data_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",   len(df))
        c2.metric("Unique Diseases", df["disease"].nunique())
        c3.metric("Unique Medicines",df["primary_medicine"].nunique())
        c4.metric("Feature Columns", df.shape[1])

        fig3 = px.histogram(df, x="disease", title="Records per Disease",
                            color="disease")
        fig3.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.pie(df, names="severity",
                      title="Severity Distribution",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Medicine Database
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Medicine Database":
    st.title("💊 Medicine Database")
    st.markdown("Explore our comprehensive medicine database containing detailed drug information, dosages, side effects, and guidelines.")
    search = st.text_input("🔍 Search Medicine", placeholder="e.g. Metformin")

    # Load directly from file to ensure newly generated dataset entries show up immediately
    # without needing to clear Streamlit's cache
    with open(os.path.join(BASE, "data", "metadata.json"), "r") as f:
        meta_data = json.load(f)
    meds = meta_data.get("medicine_info", {})

    if search:
        meds = {k: v for k, v in meds.items()
                if search.lower() in k.lower()}

    if not meds:
        st.warning("No medicines found.")
    else:
        meds_list = list(meds.items())
        items_per_page = 10
        total_items = len(meds_list)
        total_pages = (total_items - 1) // items_per_page + 1

        if 'med_page' not in st.session_state:
            st.session_state['med_page'] = 1

        if 'last_search' not in st.session_state or st.session_state['last_search'] != search:
            st.session_state['med_page'] = 1
            st.session_state['last_search'] = search

        start_idx = (st.session_state['med_page'] - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page_meds = meds_list[start_idx:end_idx]

        for name, info in current_page_meds:
            with st.expander(f"**{name}** — {info.get('drug_class','')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Generic Name:** {info.get('generic_name','–')}")
                    st.markdown(f"**Drug Class:** {info.get('drug_class','–')}")
                    st.markdown(f"**Category:** {info.get('category','–')}")
                with c2:
                    st.markdown(f"**Dosage:** {info.get('dosage','–')}")
                    se = info.get("side_effects", [])
                    st.markdown("**Side Effects:** " + (", ".join(se) if se else "–"))
                    ci = info.get("contraindications", [])
                    st.markdown("**Contraindications:** " + (", ".join(ci) if ci else "–"))

        st.divider()

        # Pagination controls moved to the bottom of the page
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1,1,2,1,1])

        with page_col1:
            if st.button("↩ Previous", use_container_width=True, disabled=(st.session_state['med_page'] <= 1)):
                st.session_state['med_page'] -= 1
                st.rerun()

        with page_col3:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; color: #555;'><b>Page {st.session_state['med_page']} of {total_pages}</b><br><small>Showing {min(items_per_page, total_items - (st.session_state['med_page']-1)*items_per_page)} of {total_items} records</small></div>", unsafe_allow_html=True)

        with page_col5:
            if st.button("Next ↪", use_container_width=True, disabled=(st.session_state['med_page'] >= total_pages)):
                st.session_state['med_page'] += 1
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Medicine Recommendation System
    **Course:** CS619 – Final Year Project | Spring 2026 \n
    **Supervisor:** Dr. Mushtaq Hussain (mushtaq.hussain@vu.edu.pk)

    ### Developer Information
    **Name:** Muhammad Ali Sanwal \n
    **VUID:** BC240440384 \n
    **Email:** bc240440384mas@vu.edu.pk \n
    **Role:** DevOps Engineer | Full-Stack Developer \n
    **Portfolio:** [https://sanwal.vercel.app/](https://sanwal.vercel.app/)

    ---
    ### Project Overview
    An AI-powered virtual assistant that provides personalized medicine recommendations
    based on patient symptoms, age, gender, and medical history.

    ### Technology Stack
    | Layer | Tools |
    |---|---|
    | Data Processing | Pandas, NumPy, Scikit-learn |
    | ML Models | Random Forest, XGBoost, LightGBM, SVM, etc. |
    | Deep Learning | TensorFlow / Keras |
    | Hyperparameter Tuning | Optuna |
    | Backend API | Flask + Flask-CORS |
    | Frontend UI | Streamlit |
    | IDE | VS Code / Google Colab |
    | Visualization | Plotly, Matplotlib, Seaborn |

    ### Architecture
    ```
    Patient Input (Symptoms + Info)
            ↓
    Feature Engineering
            ↓
    ┌──────────────────────────┐
    │  ML Disease Predictor    │
    │  ML Medicine Predictor   │
    │  Rule-Based Engine       │
    └──────────────────────────┘
            ↓
    Ranked Medicine Recommendations
            ↓
    Streamlit UI / Flask REST API
    ```

    ### Disclaimer
    This system is for **educational purposes only**. Always consult a licensed
    healthcare professional before taking any medication.
    """)
