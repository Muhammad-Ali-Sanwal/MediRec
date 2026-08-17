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
        "📋 Dataset Preview",
        "📊 Model Evaluation",
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
        <p>AI-powered personalized medicine suggestions based on patient symptoms, history & conditions</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("Patient Details, Conditions & Preferences")

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

            history = st.text_area("Medical History & Existing Conditions (optional)",
                                   placeholder="e.g., Diabetic, Hypertensive, Liver disease, Renal failure, Pregnancy...",
                                   height=80,
                                   help="System will evaluate medical history to check for drug contraindications.")

            preference = st.selectbox(
                "Medicine Category Preference (optional)",
                options=["No Preference", "OTC (Over-The-Counter) Only", "Prescription Only"],
                help="Filter or highlight medicine recommendations by category preference."
            )

            submitted = st.form_submit_button("🔍 Get Recommendations",
                                              use_container_width=True,
                                              type="primary")

    with col_result:
        if submitted:
            if not symptoms_selected:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Analyzing symptoms & safety contraindications…"):
                    result = rec.recommend(
                        symptoms_selected, age, gender, severity_val, history
                    )

                st.subheader("🎯 Diagnosis & Recommendations")

                # Predicted Diseases
                diseases = result.get("predicted_diseases", [])
                if diseases:
                    st.markdown("**Predicted Conditions (AI Model Confidence)**")
                    for d in diseases[:3]:
                        pct = d["confidence"]
                        st.markdown(f"- **{d['disease']}** — {pct:.1f}% confidence")
                        st.progress(min(pct / 100, 1.0))

                st.divider()

                # Medicine Cards
                medicines = result.get("recommended_medicines", [])
                if preference == "OTC (Over-The-Counter) Only":
                    medicines_filtered = [m for m in medicines if "OTC" in m.get("info", {}).get("category", "")]
                    medicines_to_show = medicines_filtered if medicines_filtered else medicines
                elif preference == "Prescription Only":
                    medicines_filtered = [m for m in medicines if "Prescription" in m.get("info", {}).get("category", "")]
                    medicines_to_show = medicines_filtered if medicines_filtered else medicines
                else:
                    medicines_to_show = medicines

                if medicines_to_show:
                    st.markdown("**Ranked Medicine Suggestions**")
                    for i, med in enumerate(medicines_to_show, 1):
                        info = med.get("info", {})
                        warning_text = med.get("warning")
                        
                        # Add a visual warning prefix/suffix to the card header
                        if warning_text:
                            header_title = f"⚠️ #{i} {med['medicine']} · {med.get('source','')} (Contraindicated)"
                        else:
                            header_title = f"#{i} {med['medicine']} · {med.get('source','')}"
                            
                        with st.expander(header_title):
                            if warning_text:
                                st.error(f"**⚠️ Clinical Safety Warning:** This medicine may be contraindicated. "
                                         f"Patient's medical history matches the known contraindication: **'{warning_text}'**.")
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
                st.markdown(f"**Condition Severity:** :{color}[{sev.upper()}]")

                # Confidence Chart
                if medicines_to_show:
                    conf_data = [m for m in medicines_to_show if m["confidence"] > 0]
                    if conf_data:
                        fig = go.Figure(go.Bar(
                            x=[m["medicine"] for m in conf_data],
                            y=[m["confidence"] for m in conf_data],
                            marker_color="#2980b9"
                        ))
                        fig.update_layout(
                            title="Medicine Prediction Confidence Scores (%)",
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
            st.info("Fill in the patient details on the left and click **Get Recommendations**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Dataset Preview (Requirement 1 & 2 Compliance)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Preview":
    st.title("📋 Data Collection & Preprocessing Preview")
    st.markdown("Demonstration of the clean, model-ready dataset generated and used for training the machine learning models.")

    data_path = os.path.join(BASE, "data", "patient_records.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Patient Records", f"{len(df):,}")
        m2.metric("Feature Columns", df.shape[1])
        m3.metric("Missing Values", df.isnull().sum().sum())
        m4.metric("Train / Test Split", "80% / 20%")

        st.divider()

        # Data Cleaning & Preprocessing Summary Table
        st.subheader("⚙️ Data Cleaning & Feature Engineering Summary")
        summary_table = pd.DataFrame({
            "Preprocessing Step": [
                "Data Cleaning & Imputation",
                "Text Symptom Encoding",
                "Target Variable Encoding",
                "Continuous Feature Scaling",
                "Dataset Partitioning"
            ],
            "Technique Applied": [
                "Removed duplicate records, handled missing values, noise removal",
                "One-Hot Encoding (Binary indicators for 68+ symptoms)",
                "LabelEncoder for Diseases and Primary Medicines",
                "StandardScaler (Mean=0, Std=1) for Age and Severity",
                "Stratified 80% Training / 20% Testing Split"
            ],
            "Status": ["✅ Completed", "✅ Completed", "✅ Completed", "✅ Completed", "✅ Completed"]
        })
        st.table(summary_table)

        st.divider()

        # Interactive Table Preview
        st.subheader("🔍 Interactive Dataset Viewer")
        
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            selected_disease = st.selectbox(
                "Filter by Disease",
                options=["All Diseases"] + list(df["disease"].unique())
            )
        with c_filter2:
            num_rows = st.slider("Number of Rows to Display", 5, 100, 20)

        df_filtered = df.copy()
        if selected_disease != "All Diseases":
            df_filtered = df_filtered[df_filtered["disease"] == selected_disease]

        st.dataframe(df_filtered.head(num_rows), use_container_width=True)

        st.caption(f"Showing {min(num_rows, len(df_filtered))} of {len(df_filtered)} matching records. Total dataset columns: {df.shape[1]}.")

        st.divider()

        # Visual Summaries
        st.subheader("📊 Dataset Visual Distributions")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            fig_dis = px.histogram(df, x="disease", title="Patient Records per Disease Category",
                                   color="disease", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_dis.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_dis, use_container_width=True)

        with vcol2:
            fig_sev = px.pie(df, names="severity", title="Symptom Severity Level Distribution",
                             color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig_sev, use_container_width=True)

    else:
        st.warning("Dataset file `patient_records.csv` not found. Run `python data/generate_dataset.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Training & Evaluation Results")
    st.markdown("Comparative performance evaluation across 8 Machine Learning models and 1 Deep Learning Neural Network.")

    results_path = os.path.join(BASE, "models", "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            tr = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Disease Model", tr['disease_best_model'])
        c2.metric("Best Medicine Model", tr['medicine_best_model'])
        if tr.get("deep_learning_accuracy"):
            c3.metric("Deep Learning (Keras) Accuracy", f"{tr['deep_learning_accuracy']*100:.2f}%")
        else:
            c3.metric("Evaluated Models", "8 ML + 1 DL")

        st.divider()

        # Model Comparison – Disease
        st.subheader("Disease Prediction Models Evaluation")
        disease_df = pd.DataFrame(tr["disease_results"]).T.reset_index()
        disease_df.columns = ["Model", "CV Mean", "CV Std", "Test Accuracy", "F1 Score"]
        
        st.dataframe(disease_df.style.highlight_max(axis=0, subset=["Test Accuracy", "F1 Score"], color="#d4edda"),
                     use_container_width=True)

        fig1 = px.bar(disease_df, x="Model", y="Test Accuracy",
                      color="Test Accuracy", color_continuous_scale="Blues",
                      text_auto=".3f",
                      title="Test Accuracy Comparison (Disease Prediction)")
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # Model Comparison – Medicine
        st.subheader("Medicine Prediction Models Evaluation")
        med_df = pd.DataFrame(tr["medicine_results"]).T.reset_index()
        med_df.columns = ["Model", "CV Mean", "CV Std", "Test Accuracy", "F1 Score"]

        st.dataframe(med_df.style.highlight_max(axis=0, subset=["Test Accuracy", "F1 Score"], color="#d4edda"),
                     use_container_width=True)

        fig2 = px.bar(med_df, x="Model", y="F1 Score",
                      color="F1 Score", color_continuous_scale="Greens",
                      text_auto=".3f",
                      title="Weighted F1 Score Comparison (Medicine Prediction)")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No training results found. Run `python models/train_models.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Medicine Database
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Medicine Database":
    st.title("💊 Medicine Database Reference")
    st.markdown("Explore our comprehensive medicine database containing detailed drug information, dosages, side effects, and guidelines.")
    search = st.text_input("🔍 Search Medicine", placeholder="e.g. Metformin")

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

        # Pagination controls
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
# PAGE 5 – About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Medicine Recommendation System (Prototype Phase)
    **Course:** CS619 – Final Year Project | Spring 2026 \n
    **Supervisor:** Dr. Mushtaq Hussain (mushtaq.hussain@vu.edu.pk | Teams: themushtaq48)

    ### Developer Information
    **Name:** Muhammad Ali Sanwal \n
    **VUID:** BC240440384 \n
    **Email:** bc240440384mas@vu.edu.pk \n
    **Role:** DevOps Engineer | Full-Stack Developer \n
    **Portfolio:** [https://sanwal.vercel.app/](https://sanwal.vercel.app/)

    ---
    ### Project Overview
    An AI-powered virtual assistant that provides personalized medicine recommendations
    based on patient symptoms, age, gender, severity, medical history, and existing conditions.

    ### Technology Stack
    | Layer | Tools |
    |---|---|
    | Data Processing | Pandas, NumPy, Scikit-learn |
    | ML Models | Random Forest, XGBoost, LightGBM, SVM, Logistic Regression, KNN, Naive Bayes |
    | Deep Learning | TensorFlow / Keras Neural Network |
    | Hyperparameter Tuning | Optuna |
    | Backend API | Flask + Flask-CORS |
    | Frontend UI | Streamlit |
    | Visualization | Plotly, Matplotlib, Seaborn |

    ### Architecture & Pipeline
    ```
    Patient Input (Symptoms + History + Conditions + Preferences)
            ↓
    Feature Engineering & Scaling
            ↓
    ┌──────────────────────────┐
    │  ML Disease Predictor    │
    │  ML Medicine Predictor   │
    │  Rule-Based Engine       │
    │  Safety Filter (History) │
    └──────────────────────────┘
            ↓
    Ranked Medicine Recommendations + Confidence Scores + Warnings
            ↓
    Streamlit UI / Flask REST API
    ```

    ### Disclaimer
    This system is developed for **educational and research purposes only** as part of CS619 Final Year Project. Always consult a licensed healthcare professional before taking any medication.
    """)
