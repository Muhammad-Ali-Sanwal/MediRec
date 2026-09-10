"""
MediRec Conversational AI Engine
Powered by HuggingFace Transformers, RapidFuzz Spell Checking, and Slot-Filling Dialog Management.
"""
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from src.recommender import MedicineRecommender


class MediRecChatbot:
    """
    Conversational AI Chatbot that parses natural language inputs,
    asks follow-up questions to collect missing details (slot filling),
    corrects misspellings ("Did you mean...?"), and generates exact AI recommendations.
    """

    def __init__(self, recommender: Optional[MedicineRecommender] = None):
        if recommender:
            self.recommender = recommender
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.recommender = MedicineRecommender(
                model_dir=os.path.join(base_dir, "models"),
                metadata_path=os.path.join(base_dir, "data", "metadata.json")
            )

        self._load_metadata()
        self._init_hf_model()

    def _load_metadata(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        meta_path = os.path.join(base_dir, "data", "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
            self.all_symptoms = data.get("all_symptoms", [])
            self.symptom_tags = data.get("symptom_tags", {})
        else:
            self.all_symptoms = self.recommender.all_symptoms
            self.symptom_tags = {}

        # Build alias & synonym lookup map
        self.synonym_map = {}
        for sym in self.all_symptoms:
            self.synonym_map[sym.lower()] = sym
            tag_info = self.symptom_tags.get(sym, {})
            for syn in tag_info.get("synonyms", []):
                self.synonym_map[syn.lower()] = sym

        # Add Roman Urdu medical translations
        roman_urdu_symptoms = {
            "bukhaar": "fever",
            "bukhar": "fever",
            "teez bukhar": "high fever",
            "halka bukhar": "mild fever",
            "sar dard": "headache",
            "sardard": "headache",
            "sar me dard": "headache",
            "sir dard": "headache",
            "sir me dard": "headache",
            "khansi": "cough",
            "khaansi": "cough",
            "balgham wali khansi": "cough with phlegm",
            "gala kharab": "sore throat",
            "galy me dard": "sore throat",
            "gala dard": "sore throat",
            "jism dard": "body ache",
            "jism me dard": "body ache",
            "jism pain": "body ache",
            "thakawat": "fatigue",
            "thakan": "fatigue",
            "kamzori": "fatigue",
            "sardi": "chills",
            "thabil": "chills",
            "thang lagna": "chills",
            "ulti": "vomiting",
            "ultay": "vomiting",
            "matli": "nausea",
            "dil kharab": "nausea",
            "chakkar": "dizziness",
            "sar ghoomna": "dizziness",
            "saans ki takleef": "shortness of breath",
            "sans phoolna": "shortness of breath",
            "sans ka masla": "difficulty breathing",
            "pait dard": "stomach pain",
            "pet dard": "stomach pain",
            "pait me dard": "stomach pain",
            "pait phoolna": "bloating",
            "gas": "bloating",
            "bhook na lagna": "loss of appetite",
            "tezabiyat": "burning sensation stomach",
            "seene me jalan": "burning sensation stomach",
            "peshab me jalan": "burning urination",
            "peshab ki takleef": "burning urination",
            "ziyada peshab": "frequent urination",
            "peshab frequent": "frequent urination",
            "ziyada pyas": "excessive thirst",
            "bohat pyas": "excessive thirst",
            "jodon me dard": "joint pain",
            "jod dard": "joint pain",
            "neend na aana": "difficulty falling asleep",
            "neend ka masla": "sleep problems",
            "pareshani": "excessive worry",
            "ghabrahat": "restlessness",
            "udaasi": "persistent sadness",
            "nazar dhundli": "blurred vision"
        }
        for ru_phrase, std_sym in roman_urdu_symptoms.items():
            self.synonym_map[ru_phrase] = std_sym

    def _init_hf_model(self):
        """Initialize Hugging Face zero-shot / sentence classifier pipeline safely."""
        self.hf_classifier = None
        try:
            from transformers import pipeline
            self.hf_classifier = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli"
            )
            print("✓ Loaded Hugging Face zero-shot classification pipeline.")
        except Exception as e:
            print(f"HuggingFace pipeline init notice (fallback active): {e}")

    # ── Input Extractors ──
    def extract_age(self, text: str) -> Optional[int]:
        text_lower = text.lower().strip()

        # Direct regex patterns for English and Roman Urdu age phrases
        patterns = [
            r'(?:meri umer|meri umar|meri umr|umer|umar|umr|my age is|i am|i\'m|age is|age[:=]?)\s*(?:is|=|:)?\s*(?:of)?\s*(\d{1,3})',
            r'(\d{1,3})\s*(?:saal|sal|years old|year old|years|yrs|yo)',
            r'\b(\d{1,3})\b'
        ]
        for p in patterns:
            matches = re.findall(p, text_lower)
            for match in matches:
                val = int(match)
                if 1 <= val <= 110:
                    return val
        return None

    def extract_gender(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if re.search(r'\b(female|woman|girl|lady|she|larki|aurat|female hoon|f)\b', text_lower):
            return "Female"
        if re.search(r'\b(male|man|boy|guy|he|larka|marda|male hoon|m)\b', text_lower):
            return "Male"
        if re.search(r'\b(other|non-binary)\b', text_lower):
            return "Other"
        return None

    def extract_severity(self, text: str) -> Optional[int]:
        text_lower = text.lower()
        if re.search(r'\b(serious|severe|high|critical|serous|teez|ziyada|bohat|3)\b', text_lower):
            return 3
        if re.search(r'\b(moderate|medium|average|mod|med|darmiyani|normal|2)\b', text_lower):
            return 2
        if re.search(r'\b(mild|slight|low|mid|halka|halki|kam|1)\b', text_lower):
            return 1
        return None


    def extract_medical_history(self, text: str) -> Optional[str]:
        keywords = [
            "diabetic", "diabetes", "hypertension", "high bp", "blood pressure",
            "kidney", "renal", "liver", "hepatic", "ulcer", "stomach", "pregnant",
            "pregnancy", "asthma", "heart disease", "allergy", "allergic"
        ]
        text_lower = text.lower()
        matched = []
        for kw in keywords:
            if kw in text_lower:
                matched.append(kw)
        if matched:
            return ", ".join(matched)
        return None

    def extract_symptoms(self, text: str) -> Tuple[List[str], Optional[str]]:
        """
        Extracts valid symptoms from input string.
        Returns: (matched_symptoms_list, suggestion_if_misspelled)
        """
        text_lower = text.lower()
        matched_symptoms = set()
        suggestion = None

        # 1. Direct synonym / exact phrase match
        for syn_phrase, std_symptom in self.synonym_map.items():
            if syn_phrase in text_lower:
                matched_symptoms.add(std_symptom)

        # 2. Fuzzy match for misspelled words using RapidFuzz
        if HAS_RAPIDFUZZ and not matched_symptoms:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
            ignore_words = {"have", "been", "suffering", "from", "with", "feeling", "feel", "very", "also", "some", "like", "male", "female", "years", "my", "age", "is"}
            candidate_words = [w for w in words if w not in ignore_words]

            for word in candidate_words:
                best_match = process.extractOne(
                    word,
                    list(self.synonym_map.keys()),
                    scorer=fuzz.ratio
                )
                if best_match:
                    match_str, score, _ = best_match
                    std_symptom = self.synonym_map[match_str]

                    if score >= 85:
                        matched_symptoms.add(std_symptom)
                    elif 68 <= score < 85:
                        suggestion = std_symptom
                        break

        return list(matched_symptoms), suggestion

    # ── Conversational Process ──
    def process_message(self, user_text: str, session: Dict[str, Any]) -> Dict[str, Any]:
        user_clean = user_text.strip()
        user_lower = user_clean.lower()

        # Handle pending "Did you mean...?" suggestion response
        if session.get("pending_suggestion"):
            sugg = session["pending_suggestion"]
            session["pending_suggestion"] = None
            if any(w in user_lower for w in ["yes", "yeah", "yep", "sure", "correct", "right", "ya", "y"]):
                if sugg not in session["symptoms"]:
                    session["symptoms"].append(sugg)
                return self._generate_response(session, f"✅ Got it! Added **'{sugg}'** to your symptoms.")
            elif any(w in user_lower for w in ["no", "nope", "na", "cancel"]):
                return self._generate_response(session, f"Understood. Let's skip '{sugg}'. What symptoms are you feeling?")

        # Check targeted missing slot first for intelligent auto-pick
        target_slot = None
        if session.get("symptoms") and session.get("age") is None:
            target_slot = "age"
        elif session.get("symptoms") and session.get("age") and session.get("gender") is None:
            target_slot = "gender"
        elif session.get("symptoms") and session.get("age") and session.get("gender") and session.get("severity") is None:
            target_slot = "severity"

        age = self.extract_age(user_clean)
        gender = self.extract_gender(user_clean)
        severity = self.extract_severity(user_clean)
        history = self.extract_medical_history(user_clean)
        extracted_syms, suggestion = self.extract_symptoms(user_clean)

        # Apply extracted values
        if age and not session.get("age"):
            session["age"] = age
        if gender and not session.get("gender"):
            session["gender"] = gender
        if severity and not session.get("severity"):
            session["severity"] = severity
        if history:
            if session.get("history"):
                session["history"] += ", " + history
            else:
                session["history"] = history

        for sym in extracted_syms:
            if sym not in session["symptoms"]:
                session["symptoms"].append(sym)

        # Check for initial unrelated / off-topic prompts
        greetings = {"hi", "hello", "hey", "greetings", "start", "help", "hola", "assalam", "salam"}
        is_greeting = any(g in user_lower.split() for g in greetings) or user_lower in greetings

        if not session.get("symptoms") and not extracted_syms and not suggestion and not age and not gender and not severity and not history:
            if not is_greeting:
                return {
                    "bot_message": "I didn't quite get that. I am your virtual **Medicine Recommendation Assistant** 🤖.\n\n"
                                   "Please provide your medical symptoms (e.g. *'I have a severe headache, high fever, and body ache'*), "
                                   "along with your age, gender, and severity level so I can assist you with exact recommendations!",
                    "session": session,
                    "completed": False
                }

        # Handle unrecognized responses for specific target slots
        if target_slot == "age" and session.get("age") is None:
            return {
                "bot_message": f"I didn't quite get that. I am your virtual **Medicine Recommendation Assistant** 🤖.\n\n"
                               f"Could you please tell me your **Age** (e.g. **22** or **'my age is 22'**)?",
                "session": session,
                "completed": False
            }
        if target_slot == "gender" and session.get("gender") is None:
            return {
                "bot_message": f"I didn't quite get that. I am your virtual **Medicine Recommendation Assistant** 🤖.\n\n"
                               f"Could you please tell me your **Gender** (**Male**, **Female**, or **Other**)?",
                "session": session,
                "completed": False
            }
        if target_slot == "severity" and session.get("severity") is None:
            return {
                "bot_message": f"I didn't quite get that. I am your virtual **Medicine Recommendation Assistant** 🤖.\n\n"
                               f"Could you please select your symptom severity level:\n"
                               f"• **1 (Mild)**\n• **2 (Moderate)**\n• **3 (Serious)**?",
                "session": session,
                "completed": False
            }

        # Handle spelling suggestion if detected for new symptoms
        if suggestion and suggestion not in session["symptoms"]:
            session["pending_suggestion"] = suggestion
            return {
                "bot_message": f"Did you mean **'{suggestion}'**? Please reply **Yes** or **No**.",
                "session": session,
                "completed": False
            }


        return self._generate_response(session)


    def _generate_response(self, session: Dict[str, Any], prefix_note: str = "") -> Dict[str, Any]:
        symptoms = session.get("symptoms", [])
        age = session.get("age")
        gender = session.get("gender")
        severity = session.get("severity")
        history = session.get("history", "")

        # 1. Missing Symptoms
        if not symptoms:
            msg = (prefix_note + "\n\n" if prefix_note else "") + \
                  "👋 Hello! I am **MediRec AI Assistant**.\n\n" \
                  "Please describe your symptoms (e.g. *'I have a severe headache, high fever, and body ache'*)."
            return {"bot_message": msg, "session": session, "completed": False}

        # 2. Missing Age
        if age is None:
            sym_list_str = ", ".join([f"**{s}**" for s in symptoms])
            msg = (prefix_note + "\n\n" if prefix_note else "") + \
                  f"I recorded your symptom(s): {sym_list_str}.\n\n" \
                  "Could you please tell me your **Age**?"
            return {"bot_message": msg, "session": session, "completed": False}

        # 3. Missing Gender
        if gender is None:
            msg = (prefix_note + "\n\n" if prefix_note else "") + \
                  "Got it! What is your **Gender** (Male, Female, or Other)?"
            return {"bot_message": msg, "session": session, "completed": False}

        # 4. Missing Severity
        if severity is None:
            msg = (prefix_note + "\n\n" if prefix_note else "") + \
                  "How severe are your symptoms?\n" \
                  "• **1** = Mild\n• **2** = Moderate\n• **3** = Serious"
            return {"bot_message": msg, "session": session, "completed": False}

        # All slots available! Run Recommendation Engine
        result = self.recommender.recommend(
            symptoms=symptoms,
            age=age,
            gender=gender,
            severity=severity,
            medical_history=history
        )

        return {
            "bot_message": prefix_note + "\n\n" if prefix_note else "",
            "session": session,
            "completed": True,
            "recommendation": result
        }
