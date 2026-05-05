"""
Preprocessing utilities shared across modules.
"""
import re
import nltk
import numpy as np
from typing import List

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


def normalize_symptom(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def symptoms_from_text(text: str, known_symptoms: List[str]) -> List[str]:
    """
    Extract known symptoms from free-form text input.
    Simple substring matching – good enough for this domain.
    """
    text_norm = normalize_symptom(text)
    found = []
    for sym in known_symptoms:
        if sym in text_norm:
            found.append(sym)
    return found


def severity_from_string(s: str) -> int:
    """Convert text severity to int 1-3."""
    mapping = {"mild": 1, "moderate": 2, "serious": 3, "severe": 3}
    return mapping.get(s.lower(), 2)
