"""
MediRec – ChatGPT-Style Conversational AI Assistant
Run: streamlit run src/streamlit_app.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import random
from datetime import datetime
from src.chatbot import MediRecChatbot
from src.recommender import MedicineRecommender

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRec AI – ChatGPT Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

def get_dynamic_welcome_message() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        salutation = "Good morning 🌅"
    elif 12 <= hour < 17:
        salutation = "Good afternoon 🌤️"
    elif 17 <= hour < 21:
        salutation = "Good evening 🌆"
    else:
        salutation = "Hello 🌙"

    greetings = [
        f"👋 {salutation}!\n"
        f"How are you feeling today?\n\n"
        f"I am your **Virtual Medical Assistant** 🤖. I can help in diagnosis of your disease and suggest you the medicines.\n\n"
        f"Please tell me your symptoms (in English or Roman Urdu) along with your age and gender to get started!",

        f"👋 {salutation}!\n"
        f"Welcome to **MediRec AI Assistant**.\n\n"
        f"I am your virtual medical assistant. I can assist in diagnosing your disease and recommending safe medications.\n\n"
        f"Tell me how you are feeling today (e.g. *'I am a 25 year old male experiencing fever and body ache'*).",

        f"👋 {salutation}!\n"
        f"How can I assist your health today?\n\n"
        f"I am your virtual medical assistant. I can analyze symptoms and provide personalized medicine suggestions, precautions, and dietary plans.\n\n"
        f"Please share your symptoms, age, and gender!"
    ]
    return random.choice(greetings)


# ─── Custom CSS for ChatGPT Aesthetic ──────────────────────────────────────────

st.markdown("""
<style>
    /* Dark / Light Modern ChatGPT Chat Theme */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .chat-header {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem 1rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }
    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .chat-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .sample-btn {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        cursor: pointer;
        text-align: left;
        margin-bottom: 0.5rem;
        width: 100%;
    }
    .sample-btn:hover {
        background-color: #334155;
        border-color: #38bdf8;
    }
    .card-medicine {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .disclaimer-banner {
        background-color: #451a03;
        border: 1px solid #92400e;
        color: #fef3c7;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Chatbot Engine ──────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_bot():
    rec = MedicineRecommender(
        model_dir=os.path.join(BASE, "models"),
        metadata_path=os.path.join(BASE, "data", "metadata.json")
    )
    return MediRecChatbot(recommender=rec)

bot = load_bot()

# ─── Initialize Session State ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": get_dynamic_welcome_message()
        }
    ]

if "dialog_session" not in st.session_state:
    st.session_state.dialog_session = {
        "symptoms": [],
        "age": None,
        "gender": None,
        "severity": None,
        "history": "",
        "pending_suggestion": None
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=65)
    st.title("MediRec Chatbot")
    st.caption("HuggingFace Powered AI Assistant")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": get_dynamic_welcome_message()
            }
        ]
        st.session_state.dialog_session = {
            "symptoms": [],
            "age": None,
            "gender": None,
            "severity": None,
            "history": "",
            "pending_suggestion": None
        }
        st.rerun()


    st.divider()
    st.markdown("**⚡ Quick Example Queries:**")

    sample1 = "I am a 28yo female with severe headache, nausea, and vomiting"
    sample2 = "45 year old male, frequent urination, excessive thirst, diabetic history"
    sample3 = "I have a high fever, cough, and body ache"

    if st.button("💡 " + sample1[:35] + "...", use_container_width=True):
        st.session_state.user_prompt_override = sample1

    if st.button("💡 " + sample2[:35] + "...", use_container_width=True):
        st.session_state.user_prompt_override = sample2

    if st.button("💡 " + sample3[:35] + "...", use_container_width=True):
        st.session_state.user_prompt_override = sample3

    st.divider()
    st.info("""**Developer:** M. Ali Sanwal (BC240440384)
**Course:** CS619 FYP Spring 2026
**Supervisor:** Dr. Mushtaq Hussain
**Reference Repo:** [dr-mushtaq/Medicine-Recommendation-System](https://github.com/dr-mushtaq/Medicine-Recommendation-System/)
**NLP Models:** BioBERT / ClinicalBERT / HuggingFace Transformers""")

# ─── Main Chat Interface Header ───────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🤖 MediRec ChatGPT Assistant</h1>
    <p>Conversational AI for Disease Diagnosis & Safe Medicine Recommendations<br>
    <small style="color: #38bdf8;">Powered by HuggingFace NLP · BioBERT / ClinicalBERT Architecture</small></p>
</div>
""", unsafe_allow_html=True)


def render_recommendations(rec_data: dict):
    if not rec_data:
        return

    st.divider()
    diseases = rec_data.get("predicted_diseases", [])
    medicines = rec_data.get("recommended_medicines", [])
    primary_dis = diseases[0]["disease"] if diseases else ""
    dis_info = bot.recommender.disease_details.get(primary_dis, {})

    if diseases:
        st.markdown("### 🎯 Predicted Condition(s)")
        for d in diseases[:3]:
            pct = d["confidence"]
            st.markdown(f"• **{d['disease']}** — `{pct:.1f}%` AI Confidence")
            st.progress(min(pct / 100, 1.0))

    if dis_info:
        st.markdown(f"**Description:** {dis_info.get('description', '')}")

        tab1, tab2, tab3, tab4 = st.tabs(["💊 Recommended Medicines", "🛡️ Precautions", "🥗 Dietary Plan", "🏃 Exercise & Workout"])

        with tab1:
            st.caption(f"📋 Showing medicines clinically indicated specifically for **{primary_dis}**:")
            if medicines:
                for i, med in enumerate(medicines, 1):
                    info = med.get("info", {})
                    warn = med.get("warning")

                    title = f"#{i} {med['medicine']} ({med.get('source','ML Model')})"
                    if warn:
                        title = f"⚠️ #{i} {med['medicine']} (Safety Alert)"

                    with st.expander(title, expanded=(i == 1)):
                        if warn:
                            st.error(f"**⚠️ Clinical Safety Warning:** Patient's medical history matches contraindication: **'{warn}'**.")

                        if info:
                            st.markdown(f"**Generic Name:** {info.get('generic_name','–')}")
                            st.markdown(f"**Drug Class:** {info.get('drug_class','–')}")
                            st.markdown(f"**Indication:** Clinical treatment for **{primary_dis}**")
                            st.markdown(f"**Dosage:** {info.get('dosage','–')}")
                            if info.get("side_effects"):
                                st.markdown("**Side Effects:** " + ", ".join(info["side_effects"]))
                            if info.get("contraindications"):
                                st.markdown("**Contraindications:** " + ", ".join(info["contraindications"]))
                            st.markdown(f"**Category:** `{info.get('category','–')}`")

        with tab2:
            precautions = dis_info.get("precautions", [])
            if precautions:
                for p in precautions:
                    st.markdown(f"• {p}")
            else:
                st.info("No specific precautions listed.")

        with tab3:
            diet = dis_info.get("diet", [])
            if diet:
                for item in diet:
                    st.markdown(f"• {item}")
            else:
                st.info("No specific dietary instructions.")

        with tab4:
            workout = dis_info.get("workout", [])
            if workout:
                for w in workout:
                    st.markdown(f"• {w}")
            else:
                st.info("No specific workout guidance.")

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-banner">
        ⚠️ {rec_data.get('disclaimer', '')}
    </div>
    """, unsafe_allow_html=True)


# ─── Display Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recommendation"):
            render_recommendations(msg["recommendation"])

# ─── Chat Input Handler ───────────────────────────────────────────────────────
prompt = st.chat_input("Ask MediRec AI... (e.g. 'I am 30 years old, male, experiencing high fever and body ache')")

# Check if sidebar quick sample was clicked
if "user_prompt_override" in st.session_state and st.session_state.user_prompt_override:
    prompt = st.session_state.user_prompt_override
    st.session_state.user_prompt_override = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MediRec AI is analyzing your query..."):
            response = bot.process_message(prompt, st.session_state.dialog_session)

            bot_text = response.get("bot_message", "")
            rec_result = response.get("recommendation")

            st.markdown(bot_text)

            if rec_result:
                render_recommendations(rec_result)

            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_text,
                "recommendation": rec_result
            })

