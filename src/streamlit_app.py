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


# ─── Custom CSS for Qubi & ChatGPT 4.0 Hybrid Master Theme ──────────────

st.markdown(r"""
<style>

    /* ── Root Theme Variables (Qubi / ChatGPT 4.0 Master Hybrid) ── */
    :root {
        --bg-app: #0d0d15;
        --bg-sidebar: #13131e;
        --bg-card: rgba(26, 26, 40, 0.75);
        --text-main: #f8fafc;
        --text-sub: #a1a1aa;
        --border-color: rgba(139, 92, 246, 0.2);
        --accent-violet: #8b5cf6;
        --accent-purple: #a855f7;
        --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        --user-bubble-bg: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
        --assistant-bubble-bg: rgba(28, 28, 44, 0.85);
        --input-glass-bg: rgba(24, 24, 38, 0.85);
        --input-glass-border: rgba(139, 92, 246, 0.3);
        --shadow-glow: 0 8px 32px 0 rgba(124, 58, 237, 0.25);
    }

    /* ── Light Mode Adaptive Overrides ── */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-app: #f8f9ff;
            --bg-sidebar: #f1f3f9;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --border-color: #e2e8f0;
            --accent-violet: #7c3aed;
            --accent-purple: #9333ea;
            --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            --user-bubble-bg: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
            --assistant-bubble-bg: #ffffff;
            --input-glass-bg: rgba(255, 255, 255, 0.9);
            --input-glass-border: rgba(124, 58, 237, 0.2);
            --shadow-glow: 0 8px 30px rgba(0, 0, 0, 0.08);
        }
    }

    [data-theme="light"] {
        --bg-app: #f8f9ff !important;
        --bg-sidebar: #f1f3f9 !important;
        --bg-card: #ffffff !important;
        --text-main: #0f172a !important;
        --text-sub: #64748b !important;
        --border-color: #e2e8f0 !important;
        --assistant-bubble-bg: #ffffff !important;
        --input-glass-bg: rgba(255, 255, 255, 0.9) !important;
    }

    /* Global App Background */
    .stApp {
        background-color: var(--bg-app) !important;
        background-image: radial-gradient(circle at 50% -20%, rgba(139, 92, 246, 0.15), transparent 70%) !important;
        color: var(--text-main) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Qubi Hero Banner */
    .qubi-hero-card {
        text-align: center;
        padding: 2.2rem 1rem 1.2rem 1rem;
        background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.15), transparent 70%);
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }

    .qubi-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #c084fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .qubi-hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text-main);
        letter-spacing: -0.8px;
        line-height: 1.25;
        margin-bottom: 0.6rem;
    }

    .gradient-text {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .qubi-hero-sub {
        color: var(--text-sub);
        font-size: 1rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Action Cards Styling */
    .action-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        height: 100%;
    }

    .action-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-violet) !important;
        box-shadow: var(--shadow-glow) !important;
    }

    .action-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }

    .action-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text-main);
    }

    .action-card-arrow {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: rgba(139, 92, 246, 0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #c084fc;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ─── Sidebar Recent Chats Hover Reveal & Full Width Layout ─── */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        position: relative !important;
        align-items: center !important;
        border-radius: 10px !important;
        transition: background-color 0.2s ease !important;
        margin-bottom: 2px !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        background-color: rgba(139, 92, 246, 0.08) !important;
    }

    /* Title Button - Full Width & Clean Text */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(1) {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        text-align: left !important;
        justify-content: flex-start !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
        padding-left: 10px !important;
        border-radius: 8px !important;
    }

    /* Action Buttons (Edit & Delete) - Hidden by default */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) {
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        transition: opacity 0.2s ease, visibility 0.2s ease !important;
        flex: 0 0 auto !important;
    }

    /* Reveal Action Buttons ONLY on Hover of the Chat Row (or always when editing) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover > div:nth-child(2),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover > div:nth-child(3),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(input) > div:nth-child(2),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(input) > div:nth-child(3) {
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }


    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        padding: 0 !important;
        margin: 0 auto !important;
        font-size: 0.95rem !important;
        line-height: 1 !important;
        border-radius: 6px !important;
        text-align: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button *,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button * {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
        text-align: center !important;
        width: 100% !important;
        height: 100% !important;
    }


    /* Card & Medicine Styling */
    .card-medicine {

        background: var(--bg-card) !important;
        border-left: 4px solid var(--accent-violet) !important;
        border-right: 1px solid var(--border-color) !important;
        border-top: 1px solid var(--border-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding: 1.2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-glow);
        color: var(--text-main) !important;
    }

    /* Chat Messages - ChatGPT 4.0 Style */
    [data-testid="stChatMessage"] {
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
        border-radius: 18px !important;
    }

    /* User Chat Bubble - Purple Gradient */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--user-bubble-bg) !important;
        color: #ffffff !important;
        border-radius: 20px 20px 4px 20px !important;
        margin-left: 2rem !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-size: 1.02rem !important;
    }

    /* Assistant Chat Bubble - Dark Glass */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--assistant-bubble-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 4px solid var(--accent-violet) !important;
        border-radius: 20px 20px 20px 4px !important;
        margin-right: 2rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2) !important;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.15) !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        color: #f59e0b !important;
        padding: 0.9rem;
        border-radius: 12px;
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
        background-color: var(--accent-violet);
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Floating Translucent Pill Chat Input Bar */
    .stChatInputContainer {
        padding-bottom: 1.5rem !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        background: var(--input-glass-bg) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid var(--input-glass-border) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37) !important;
        padding: 4px 14px !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-violet) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3), 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    /* Remove inner rectangular borders & box shadows from BaseWeb elements */
    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }

    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: var(--text-main) !important;
        font-size: 1rem !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }

    [data-testid="stChatInputTextArea"]:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        border-radius: 50% !important;
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        border: none !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    [data-testid="stChatInputSubmitButton"]:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.6) !important;
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
                col_inp, col_save, col_canc = st.columns([7.5, 1.25, 1.25])
                with col_inp:
                    new_t = st.text_input("Rename", value=s_title, key=f"edit_input_{s_id}", label_visibility="collapsed")
                with col_save:
                    if st.button("✓", key=f"save_title_{s_id}", help="Save Title", type="primary"):
                        if new_t.strip():
                            cs.rename_session(s_id, new_t)
                            if is_active:
                                st.session_state.active_session_title = new_t.strip()
                        st.session_state.editing_session_id = None
                        st.rerun()
                with col_canc:
                    if st.button("✖", key=f"cancel_title_{s_id}", help="Cancel"):
                        st.session_state.editing_session_id = None
                        st.rerun()

            else:
                col_title, col_edit, col_del = st.columns([8.2, 0.9, 0.9])

                with col_title:
                    clean_t = s_title.replace("💬", "").replace("👉", "").strip()

                    if st.button(clean_t, key=f"select_chat_{s_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                        if not is_active:
                            switch_active_session(s)
                            st.rerun()

                with col_edit:
                    if st.button("✎", key=f"btn_edit_{s_id}", help="Rename chat title"):
                        st.session_state.editing_session_id = s_id
                        st.rerun()

                with col_del:
                    if st.button("✖", key=f"btn_del_{s_id}", help="Delete chat"):
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
            st.warning("⚠️ PDF generator building... Use text download below.")

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
        ⚠️ {rec_data.get('disclaimer', '')}
    </div>
    """, unsafe_allow_html=True)


# ─── Display Qubi Hero & Action Cards (New Chat State) ───────────────────────
has_user_messages = any(m.get("role") == "user" for m in st.session_state.messages)

if not has_user_messages:
    st.markdown("""
    <div class="qubi-hero-card">
        <div class="qubi-tag">🤖 BioBERT & Clinical AI Engine</div>
        <h1 class="qubi-hero-title">How can we <span class="gradient-text">assist</span> your health today?</h1>
        <p class="qubi-hero-sub">Get personalized clinical guidance powered by medical NLP models. Select an example topic below or describe your symptoms in the chatbox.</p>
    </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🩺 **Symptom Diagnostics**\n\n*I am a 28yo female with severe headache, nausea, and vomiting*", key="card1", use_container_width=True):
            st.session_state.user_prompt_override = "I am a 28yo female with severe headache, nausea, and vomiting"
            st.rerun()

        if st.button("🤒 **Fever & Acute Symptoms**\n\n*I have a high fever, cough, and body ache*", key="card3", use_container_width=True):
            st.session_state.user_prompt_override = "I have a high fever, cough, and body ache"
            st.rerun()

    with col_c2:
        if st.button("💊 **Diabetic & Chronic Care**\n\n*45 year old male, frequent urination, diabetic history*", key="card2", use_container_width=True):
            st.session_state.user_prompt_override = "45 year old male, frequent urination, excessive thirst, diabetic history"
            st.rerun()

        if st.button("🛡️ **Medication & Safety**\n\n*What are the side effects and dosage of Paracetamol?*", key="card4", use_container_width=True):
            st.session_state.user_prompt_override = "What are the side effects and dosage of Paracetamol?"
            st.rerun()
    st.divider()

# ─── Display Chat History ──────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recommendation"):
            render_recommendations(msg["recommendation"], key_prefix=f"hist_{idx}")

# ─── Chat Input Handler ───────────────────────────────────────────────────────
prompt = st.chat_input("Type your symptoms or health question here...")

if "user_prompt_override" in st.session_state and st.session_state.user_prompt_override:
    prompt = st.session_state.user_prompt_override
    st.session_state.user_prompt_override = None


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        import time
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

# ─── Browser localStorage Sync Bridge ──────────────────────────────────────────
try:
    s_id = st.session_state.active_session_id
    s_title = st.session_state.active_session_title
    s_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    payload = json.dumps({"id": s_id, "title": s_title, "messages": s_msgs})
    st.markdown(f"""
    <script>
    try {{
        const activeData = {payload};
        localStorage.setItem('medirec_active_chat', JSON.stringify(activeData));
        let allChats = JSON.parse(localStorage.getItem('medirec_all_chats') || '{{}}');
        allChats[activeData.id] = activeData;
        localStorage.setItem('medirec_all_chats', JSON.stringify(allChats));
    }} catch (e) {{
        console.error('localStorage sync error:', e);
    }}
    </script>
    """, unsafe_allow_html=True)
except Exception:
    pass




