"""
Chat Storage Manager for MediRec Application.
Handles loading, saving, renaming, and deleting persistent chat sessions from data/chat_history.json.
"""
import os
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_PATH = os.path.join(BASE_DIR, "data", "chat_history.json")


def load_all_sessions() -> List[Dict[str, Any]]:
    """Loads all chat sessions from JSON storage file, sorted by last updated timestamp."""
    if not os.path.exists(STORAGE_PATH):
        return []
    try:
        with open(STORAGE_PATH, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    except Exception:
        return []


def save_all_sessions(sessions: List[Dict[str, Any]]):
    """Saves list of chat sessions to JSON storage file."""
    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
    with open(STORAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific chat session by ID."""
    sessions = load_all_sessions()
    for s in sessions:
        if s.get("id") == session_id:
            return s
    return None


def create_or_update_session(session_id: str, title: str, messages: List[Dict[str, Any]], dialog_session: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new session or updates an existing session."""
    sessions = load_all_sessions()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing = None
    for s in sessions:
        if s.get("id") == session_id:
            existing = s
            break

    if existing:
        if title and title != "New Chat":
            existing["title"] = title
        existing["messages"] = messages
        existing["dialog_session"] = dialog_session
        existing["updated_at"] = now_str
        target = existing
    else:
        target = {
            "id": session_id or str(uuid.uuid4()),
            "title": title or "New Chat",
            "created_at": now_str,
            "updated_at": now_str,
            "messages": messages,
            "dialog_session": dialog_session
        }
        sessions.append(target)

    save_all_sessions(sessions)
    return target


def rename_session(session_id: str, new_title: str) -> bool:
    """Renames a specific chat session."""
    sessions = load_all_sessions()
    for s in sessions:
        if s.get("id") == session_id:
            s["title"] = new_title.strip()
            s["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_all_sessions(sessions)
            return True
    return False


def delete_session(session_id: str) -> bool:
    """Deletes a specific chat session."""
    sessions = load_all_sessions()
    filtered = [s for s in sessions if s.get("id") != session_id]
    if len(filtered) < len(sessions):
        save_all_sessions(filtered)
        return True
    return False
