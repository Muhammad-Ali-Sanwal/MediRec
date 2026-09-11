"""
MediRec – AI Assistant
Run: streamlit run src/streamlit_app.py
"""
import os
import sys
import json
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import random
import random
from datetime import datetime
from src.chatbot import MediRecChatbot
from src.recommender import MedicineRecommender
import src.chat_storage as cs

try:
    from src.pdf_generator import generate_pdf_report, HAS_REPORTLAB
except ImportError:
    HAS_REPORTLAB = False
    generate_pdf_report = None

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRec AI –  Assistant",
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


# ─── Custom CSS for ChatGPT Glassmorphic & Light/Dark Theme Aesthetic ───────

st.markdown("""
<style>
    /* ── Root Theme Variables (Default Dark Mode) ── */
    :root {
        --bg-app: #0f172a;
        --bg-sidebar: #1e293b;
        --bg-card: #1e293b;
        --text-main: #f8fafc;
        --text-sub: #94a3b8;
        --border-color: #334155;
        --accent-color: #38bdf8;
        --accent-hover: #0284c7;
        --input-glass-bg: rgba(30, 41, 59, 0.75);
        --input-glass-border: rgba(255, 255, 255, 0.15);
        --input-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    /* ── Light Theme Adaptive Overrides ── */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-app: #f8fafc;
            --bg-sidebar: #f1f5f9;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-sub: #475569;
            --border-color: #cbd5e1;
            --accent-color: #0284c7;
            --accent-hover: #0369a1;
            --input-glass-bg: rgba(255, 255, 255, 0.85);
            --input-glass-border: rgba(0, 0, 0, 0.12);
            --input-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }
    }

    /* Streamlit explicit light mode selector */
    [data-theme="light"] {
        --bg-app: #f8fafc !important;
        --bg-sidebar: #f1f5f9 !important;
        --bg-card: #ffffff !important;
        --text-main: #0f172a !important;
        --text-sub: #475569 !important;
        --border-color: #cbd5e1 !important;
        --accent-color: #0284c7 !important;
        --accent-hover: #0369a1 !important;
        --input-glass-bg: rgba(255, 255, 255, 0.85) !important;
        --input-glass-border: rgba(0, 0, 0, 0.12) !important;
        --input-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
    }

    /* Global App Container */
    .stApp {
        background-color: var(--bg-app) !important;
        color: var(--text-main) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Header Section */
    .chat-header {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem 1rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-color) 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .chat-header p {
        color: var(--text-sub);
        font-size: 0.95rem;
    }

    /* Card & Medicine Styling */
    .card-medicine {
        background: var(--bg-card) !important;
        border-left: 4px solid var(--accent-color) !important;
        border-right: 1px solid var(--border-color) !important;
        border-top: 1px solid var(--border-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: var(--input-shadow);
        color: var(--text-main) !important;
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--text-main) !important;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.15) !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        color: #d97706 !important;
        padding: 0.9rem;
        border-radius: 10px;
        font-size: 0.88rem;
        margin-top: 1.2rem;
    }

    /* Pulsing 3-Dots Typing Animation */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-color);
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* ─── Sleek & Modern ChatGPT Input Bar ─── */
    .stChatInputContainer {
        padding-bottom: 1.5rem !important;
        background-color: transparent !important;
    }

    /* Outer Capsule Wrapper */
    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        background-color: #212124 !important;
        border: 1px solid #38383e !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        padding: 4px 12px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* Focus & Hover States */
    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"]:hover {
        border-color: #52525b !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
    }

    /* Remove inner red/white outlines from BaseWeb textarea containers */
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Text Area */
    [data-testid="stChatInputTextArea"] {
        background-color: transparent !important;
        color: #f4f4f5 !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        font-size: 0.98rem !important;
    }

    [data-testid="stChatInputTextArea"]:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Placeholder text */
    [data-testid="stChatInputTextArea"]::placeholder {
        color: #a1a1aa !important;
    }

    /* Send Button */
    [data-testid="stChatInputSubmitButton"] {
        border-radius: 50% !important;
        background-color: #3f3f46 !important;
        color: #ffffff !important;
        border: none !important;
        width: 32px !important;
        height: 32px !important;
        transition: background-color 0.2s ease, transform 0.2s ease !important;
    }

    [data-testid="stChatInputSubmitButton"]:hover {
        background-color: #ffffff !important;
        color: #09090b !important;
        transform: scale(1.05) !important;
    }

    /* ── Light Theme Adaptations ── */
    @media (prefers-color-scheme: light) {
        [data-testid="stChatInput"] {
            background-color: #f4f4f6 !important;
            border: 1px solid #e4e4e7 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
        }

        [data-testid="stChatInputTextArea"] {
            color: #18181b !important;
        }

        [data-testid="stChatInputTextArea"]::placeholder {
            color: #71717a !important;
        }

        [data-testid="stChatInputSubmitButton"] {
            background-color: #18181b !important;
            color: #ffffff !important;
        }

        [data-testid="stChatInputSubmitButton"]:hover {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
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

# ─── Initialize Session State & Persistence ──────────────────────────────────
import uuid

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = str(uuid.uuid4())

if "active_session_title" not in st.session_state:
    st.session_state.active_session_title = "New Chat"

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

if "pending_delete_id" not in st.session_state:
    st.session_state.pending_delete_id = None

if "editing_session_id" not in st.session_state:
    st.session_state.editing_session_id = None


def generate_smart_chat_title(prompt: str, rec_result: dict = None, symptoms: list = None) -> str:
    """Generates clean, emoji-free chat session titles based on disease prediction, symptoms, or prompt context."""
    # 1. If prediction result exists with disease
    if rec_result and rec_result.get("predicted_diseases"):
        top_disease = rec_result["predicted_diseases"][0]["disease"]
        clean_d = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]', '', top_disease).strip()
        if clean_d:
            return f"{clean_d.title()} Assistance"

    # 2. If symptoms list exists
    if symptoms and len(symptoms) > 0:
        sym_str = " ".join([str(s).replace("_", " ") for s in symptoms[:2]]).title()
        return f"{sym_str} Assistance"

    # 3. Derive from prompt text
    clean_p = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]', '', prompt).strip()
    words = [w for w in clean_p.split() if len(w) > 2 and w.lower() not in {"this", "that", "with", "have", "from", "your", "what", "year", "years", "male", "female", "old"}]
    if len(words) >= 2:
        title_base = " ".join(words[:3]).title()
    elif len(words) == 1:
        title_base = words[0].title()
    else:
        title_base = clean_p[:20].title() if clean_p else "Medical"

    title_base = title_base.replace("Assistance", "").replace("Chat", "").strip()
    return f"{title_base} Assistance" if title_base else "Medical Assistance"


def switch_active_session(session_data: dict):

    st.session_state.active_session_id = session_data["id"]
    st.session_state.active_session_title = session_data.get("title", "New Chat")
    st.session_state.messages = session_data.get("messages", [])
    st.session_state.dialog_session = session_data.get("dialog_session", {
        "symptoms": [], "age": None, "gender": None, "severity": None, "history": "", "pending_suggestion": None
    })
    st.session_state.pending_delete_id = None
    st.session_state.editing_session_id = None

def start_new_chat():
    if any(m.get("role") == "user" for m in st.session_state.messages):
        cs.create_or_update_session(
            st.session_state.active_session_id,
            st.session_state.active_session_title,
            st.session_state.messages,
            st.session_state.dialog_session
        )
    st.session_state.active_session_id = str(uuid.uuid4())
    st.session_state.active_session_title = "New Chat"
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": get_dynamic_welcome_message()
        }
    ]
    st.session_state.dialog_session = {
        "symptoms": [], "age": None, "gender": None, "severity": None, "history": "", "pending_suggestion": None
    }
    st.session_state.pending_delete_id = None
    st.session_state.editing_session_id = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=60)
    st.title("MediRec AI Assistant")
    st.caption("Healthcare Conversational Assistant")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()

    # Load persistent chat sessions
    all_sessions = cs.load_all_sessions()

    # Delete Confirmation Warning Modal
    if st.session_state.pending_delete_id:
        p_id = st.session_state.pending_delete_id
        target_s = cs.get_session(p_id)
        t_name = target_s.get("title", "this chat") if target_s else "this chat"

        st.warning(f"⚠️ Delete **'{t_name[:22]}'**?")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("🗑️ Yes", use_container_width=True, type="primary", key="confirm_del_yes"):
                cs.delete_session(p_id)
                if st.session_state.active_session_id == p_id:
                    start_new_chat()
                else:
                    st.session_state.pending_delete_id = None
                st.rerun()
        with col_del2:
            if st.button("❌ Cancel", use_container_width=True, key="confirm_del_no"):
                st.session_state.pending_delete_id = None
                st.rerun()
        st.divider()

    st.markdown("**RECENT CHATS**")

    if not all_sessions:
        st.caption("No previous chats saved yet.")
    else:
        for idx, s in enumerate(all_sessions):
            s_id = s["id"]
            s_title = s.get("title", "New Chat")
            is_active = (s_id == st.session_state.active_session_id)

            if st.session_state.editing_session_id == s_id:
                new_t = st.text_input("Rename title:", value=s_title, key=f"edit_input_{s_id}")
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    if st.button("Save", key=f"save_title_{s_id}", use_container_width=True, type="primary"):
                        if new_t.strip():
                            cs.rename_session(s_id, new_t)
                            if is_active:
                                st.session_state.active_session_title = new_t.strip()
                        st.session_state.editing_session_id = None
                        st.rerun()
                with col_e2:
                    if st.button("Cancel", key=f"cancel_title_{s_id}", use_container_width=True):
                        st.session_state.editing_session_id = None
                        st.rerun()
            else:
                col_title, col_edit, col_del = st.columns([5.5, 1.2, 1.2])

                with col_title:
                    clean_t = s_title.replace("💬", "").replace("👉", "").strip()
                    btn_label = f"{clean_t[:22]}..." if len(clean_t) > 22 else clean_t

                    if st.button(btn_label, key=f"select_chat_{s_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                        if not is_active:
                            switch_active_session(s)
                            st.rerun()

                with col_edit:
                    if st.button("✏️", key=f"btn_edit_{s_id}", help="Rename chat title"):
                        st.session_state.editing_session_id = s_id
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"btn_del_{s_id}", help="Delete chat"):
                        st.session_state.pending_delete_id = s_id
                        st.rerun()



def render_recommendations(rec_data: dict, key_prefix: str = ""):
    if not rec_data:
        return

    if not key_prefix:
        import uuid
        key_prefix = str(uuid.uuid4())[:8]

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
                        title = f"\u26a0\ufe0f #{i} {med['medicine']} (Safety Alert)"

                    with st.expander(title, expanded=(i == 1)):
                        if warn:
                            st.error(f"**\u26a0\ufe0f Clinical Safety Warning:** Patient's medical history matches contraindication: **'{warn}'**.")

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

    # Download Consultation Report Buttons
    session_data = st.session_state.get("dialog_session", {})
    
    col_pdf, col_txt = st.columns([2, 1])
    
    with col_pdf:
        if HAS_REPORTLAB and generate_pdf_report is not None:
            try:
                pdf_bytes = generate_pdf_report(session_data, rec_data, bot.recommender.disease_details)
                st.download_button(
                    label="Download PDF Prescription (.pdf)",
                    data=pdf_bytes,
                    file_name=f"MediRec_Prescription_{primary_dis.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key=f"pdf_btn_{key_prefix}"
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
        else:
            st.warning("\u26a0\ufe0f PDF generator building... Use text download below.")

    with col_txt:
        report_text = bot.generate_consultation_report(session_data, rec_data)
        st.download_button(
            label="Download Text (.txt)",
            data=report_text,
            file_name=f"MediRec_Consultation_Report_{primary_dis.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"txt_btn_{key_prefix}"
        )

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-banner">
        \u26a0\ufe0f {rec_data.get('disclaimer', '')}
    </div>
    """, unsafe_allow_html=True)


# ─── Display Chat History ──────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recommendation"):
            render_recommendations(msg["recommendation"], key_prefix=f"hist_{idx}")

# ─── Chat Input Handler ───────────────────────────────────────────────────────
prompt = st.chat_input("Ask MediRec AI... (e.g. 'I am 30 years old, male, experiencing high fever and body ache')")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        import time
        # Dynamic random delay between 400 ms and 1100 ms (e.g. 500 ms, 800 ms, 1000 ms randomly)
        random_thinking_delay = random.uniform(0.4, 1.1)

        with st.spinner("MediRec AI is analyzing your query... ● ● ●"):
            time.sleep(random_thinking_delay)
            response = bot.process_message(prompt, st.session_state.dialog_session)

            bot_text = response.get("bot_message", "")
            rec_result = response.get("recommendation")

            st.markdown(bot_text)

            if rec_result:
                render_recommendations(rec_result, key_prefix=f"new_{len(st.session_state.messages)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_text,
                "recommendation": rec_result
            })

            # Auto update clean chat title based on prediction/chat context (no emojis)
            extracted_symptoms = st.session_state.dialog_session.get("symptoms", [])
            new_title = generate_smart_chat_title(prompt, rec_result, extracted_symptoms)

            # Auto-update if it's currently 'New Chat' OR if a disease prediction was just made
            if st.session_state.active_session_title == "New Chat" or (rec_result and rec_result.get("predicted_diseases")):
                st.session_state.active_session_title = new_title

            # Save session to persistent storage
            cs.create_or_update_session(
                st.session_state.active_session_id,
                st.session_state.active_session_title,
                st.session_state.messages,
                st.session_state.dialog_session
            )

            st.rerun()



