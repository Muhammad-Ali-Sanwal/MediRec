"""
MediRec Conversational AI Engine
Powered by HuggingFace Transformers, RapidFuzz Spell Checking, and Slot-Filling Dialog Management.
"""
import os
import re
import json
import random
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

        # Add English natural language synonyms & colloquial health terms
        english_symptom_synonyms = {
            # Eye & Headache terms
            "pain behind the eyes": ["severe headache", "light sensitivity"],
            "pain behind eyes": ["severe headache", "light sensitivity"],
            "pain behind eye": ["severe headache"],
            "pain in eyes": ["light sensitivity", "headache"],
            "eye pain": ["light sensitivity", "headache"],
            "pain in back of eyes": ["severe headache"],
            "head pain": "headache",
            "head ache": "headache",

            # Flu & Infection terms
            "flu": ["fever", "body ache", "cough", "chills"],
            "flue": ["fever", "body ache", "cough"],
            "influenza": ["fever", "body ache", "cough", "chills"],
            "cold": ["cough", "runny nose", "sore throat"],
            "common cold": ["cough", "runny nose", "sore throat"],
            "feverish": "fever",
            "high temperature": "high fever",
            "body pain": "body ache",
            "pain in body": "body ache",
            "body hurts": "body ache",
            "throat pain": "sore throat",
            "pain in throat": "sore throat",
            "scratchy throat": "sore throat",
            "stomach pain": "stomach pain",
            "stomach ache": "stomach pain",
            "pain in stomach": "stomach pain",
            "belly pain": "stomach pain",
            "vomit": "vomiting",
            "vomitting": "vomiting",
            "throwing up": "vomiting",
            "nauseous": "nausea",
            "feeling sick": "nausea",
            "dizzy": "dizziness",
            "spinning": "dizziness",
            "tired": "fatigue",
            "tiredness": "fatigue",
            "exhausted": "fatigue",
            "short of breath": "shortness of breath",
            "breathless": "shortness of breath",
            "trouble breathing": "difficulty breathing",
            "breathing difficulty": "difficulty breathing",
            "chest pain": "chest pain",
            "pain in chest": "chest pain",
            "joint pain": "joint pain",
            "pain in joints": "joint pain",
            "burning when peeing": "burning urination",
            "painful urination": "burning urination",
            "urination pain": "burning urination",
            "frequent peeing": "frequent urination",
            "peeing a lot": "frequent urination",
            "thirsty": "excessive thirst",
            "very thirsty": "excessive thirst",
            "blurred vision": "blurred vision",
            "blurry vision": "blurred vision",
            "can't sleep": "difficulty falling asleep",
            "sleeplessness": "difficulty falling asleep",
            "insomnia": "difficulty falling asleep"
        }

        for en_phrase, std_sym in english_symptom_synonyms.items():
            self.synonym_map[en_phrase] = std_sym

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

        # 1. Direct synonym / exact phrase match (sorted by length descending for multi-word phrases)
        sorted_phrases = sorted(self.synonym_map.keys(), key=lambda x: len(x), reverse=True)
        for syn_phrase in sorted_phrases:
            if syn_phrase in text_lower:
                std_val = self.synonym_map[syn_phrase]
                if isinstance(std_val, list):
                    for s in std_val:
                        matched_symptoms.add(s)
                else:
                    matched_symptoms.add(std_val)

        # 2. Direct disease entity recognition fallback (e.g. 'flu', 'migraine', 'cold')
        if not matched_symptoms:
            for dis_name, dis_data in self.recommender.disease_details.items():
                if dis_name.lower() in text_lower:
                    for s in dis_data.get("symptoms", []):
                        matched_symptoms.add(s)

        # 3. Fuzzy match for misspelled words using RapidFuzz
        if HAS_RAPIDFUZZ and not matched_symptoms:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
            ignore_words = {"have", "been", "suffering", "from", "with", "feeling", "feel", "very", "also", "some", "like", "male", "female", "years", "my", "age", "is"}
            candidate_words = [w for w in words if w not in ignore_words]

            string_synonyms = [k for k, v in self.synonym_map.items() if isinstance(v, str)]
            for word in candidate_words:
                best_match = process.extractOne(
                    word,
                    string_synonyms,
                    scorer=fuzz.ratio
                )
                if best_match:
                    match_str, score, _ = best_match
                    std_symptom = self.synonym_map[match_str]
                    if isinstance(std_symptom, str):
                        if score >= 85:
                            matched_symptoms.add(std_symptom)
                        elif 68 <= score < 85:
                            suggestion = std_symptom
                            break

        return list(matched_symptoms), suggestion


    def _handle_general_medical_query(self, user_clean: str) -> Optional[Dict[str, Any]]:
        """
        Intelligently answers general medical, drug, disease, diet, or safety questions.
        """
        user_lower = user_clean.lower()
        med_info_map = self.recommender.medicine_info
        dis_info_map = self.recommender.disease_details

        # 1. Check if asking about a specific medicine
        matched_med = None
        for med_name in med_info_map.keys():
            if med_name.lower() in user_lower:
                matched_med = med_name
                break

        if matched_med:
            info = med_info_map[matched_med]
            thinking = [
                f"• **Step 1:** Identified target pharmacological entity: **'{matched_med}'**.",
                f"• **Step 2:** Queried clinical metadata database (`metadata.json`).",
                f"• **Step 3:** Extracted Generic Name, Drug Class (`{info.get('drug_class')}`), Dosage, Side Effects, and Contraindications.",
                f"• **Step 4:** Formulated comprehensive clinical medical overview."
            ]

            ans = f"### 💊 Clinical Overview: **{matched_med}**\n\n" \
                  f"• **Generic Name:** {info.get('generic_name', 'N/A')}\n" \
                  f"• **Drug Class:** {info.get('drug_class', 'N/A')}\n" \
                  f"• **Category:** `{info.get('category', 'Prescription')}`\n" \
                  f"• **Typical Dosage:** {info.get('dosage', 'N/A')}\n\n"

            if info.get("side_effects"):
                ans += f"**Common Side Effects:**\n" + "\n".join([f"• {se}" for se in info["side_effects"]]) + "\n\n"
            if info.get("contraindications"):
                ans += f"**Contraindications & Warnings:**\n" + "\n".join([f"• ⚠️ {ci}" for ci in info["contraindications"]]) + "\n\n"

            ans += "*Always consult a licensed medical doctor before administering any medication.*"
            return {"bot_message": ans, "thinking": thinking}

        # 2. Check if asking about a specific disease / condition
        matched_dis = None
        for dis_name in dis_info_map.keys():
            if dis_name.lower() in user_lower:
                matched_dis = dis_name
                break

        if matched_dis:
            d_info = dis_info_map[matched_dis]
            thinking = [
                f"• **Step 1:** Identified medical condition entity: **'{matched_dis}'**.",
                f"• **Step 2:** Cross-referenced clinical guidelines & symptom database.",
                f"• **Step 3:** Extracted Description, Symptoms, Indicated Medications, Precautions, and Diet.",
                f"• **Step 4:** Formulated clinical disease overview."
            ]

            ans = f"### 🩺 Clinical Overview: **{matched_dis}**\n\n" \
                  f"**Description:** {d_info.get('description', '')}\n\n" \
                  f"**Common Symptoms:** {', '.join(d_info.get('symptoms', []))}\n\n" \
                  f"**Clinically Indicated Medications:** {', '.join(d_info.get('medicines', []))}\n\n"

            if d_info.get("precautions"):
                ans += f"**🛡️ Key Precautions:**\n" + "\n".join([f"• {p}" for p in d_info["precautions"]]) + "\n\n"
            if d_info.get("diet"):
                ans += f"**🥗 Recommended Diet:**\n" + "\n".join([f"• {dt}" for dt in d_info["diet"]]) + "\n\n"

            return {"bot_message": ans, "thinking": thinking}

        return None

    # ── Conversational Process ──
    def process_message(self, user_text: str, session: Dict[str, Any]) -> Dict[str, Any]:
        user_clean = user_text.strip()
        user_lower = user_clean.lower()

        # Check for general medical queries first
        gen_res = self._handle_general_medical_query(user_clean)
        if gen_res and not session.get("symptoms"):
            gen_res["session"] = session
            gen_res["completed"] = False
            return gen_res

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
                                   "or ask me any health/drug question (e.g. *'What are the side effects of Paracetamol?'*)!",
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

        # 4. Severity (Default to 2 - Moderate if age and gender are present)
        if severity is None:
            severity = 2
            session["severity"] = 2


        # All slots available! Run Recommendation Engine
        result = self.recommender.recommend(
            symptoms=symptoms,
            age=age,
            gender=gender,
            severity=severity,
            medical_history=history
        )

        top_dis = result.get("predicted_diseases", [{}])[0].get("disease", "Unknown Condition")
        top_conf = result.get("predicted_diseases", [{}])[0].get("confidence", 0.0)

        thinking = [
            f"• **Step 1: Clinical Data Collection:** Symptoms = `{', '.join(symptoms)}`, Age = `{age}`, Gender = `{gender}`, Severity = `{severity}`.",
            f"• **Step 2: BioBERT / NLP Vector Mapping:** Processed free-form input & mapped terms against symptom database.",
            f"• **Step 3: Multi-Class Disease Risk Evaluation:** Predicted top candidate **'{top_dis}'** with **{top_conf:.1f}% AI confidence**.",
            f"• **Step 4: Guideline Indication Filtering:** Verified ML predictions against `allowed_meds` clinical indication list.",
            f"• **Step 5: Safety & Contraindication Cross-Check:** Matched medical history (`{history or 'None'}`) against drug contraindications.",
            f"• **Step 6: Plan Synthesis:** Formulated prescription, precautions, dietary, and exercise advice."
        ]

        # Dynamic context-aware clinical summary generator
        sev_labels = {1: "Mild", 2: "Moderate", 3: "Serious"}
        sev_name = sev_labels.get(severity, "Moderate")
        sym_str = ", ".join([f"**{s}**" for s in symptoms])
        hist_note = f" (considering your history of **{history}**)" if history else ""

        phrasings = [
            f"✅ **Got it!** Based on your reported symptoms ({sym_str}) for a **{age}-year-old {gender.lower()}** with **{sev_name.lower()}** severity{hist_note}, I have evaluated your profile against our BioBERT clinical database. Here are the top predicted condition (**{top_dis}** at **{top_conf:.1f}% confidence**) and recommended treatment plan:",

            f"✅ **Thank you for providing all details.** Analyzing your presentation of {sym_str} in a **{age}yo {gender.lower()}** ({sev_name.lower()} severity){hist_note}: Our AI models indicate **{top_dis}** as the primary diagnosis. Below is your personalized medicine, precaution, and dietary plan:",

            f"✅ **Clinical Evaluation Complete!** For a **{age}-year-old {gender.lower()}** presenting with {sym_str} ({sev_name.lower()} severity){hist_note}, our Hugging Face AI pipeline has matched your symptoms to **{top_dis}** ({top_conf:.1f}% confidence). Here are your tailored medical recommendations:",

            f"✅ **Understood.** I have processed your complete profile — **{age}yo {gender.lower()}**, {sev_name.lower()} severity, symptoms: {sym_str}{hist_note}. The primary clinical finding is **{top_dis}**. Below are the recommended medications, precautions, and lifestyle guidance:"
        ]

        dynamic_summary = random.choice(phrasings)
        final_msg = (prefix_note + "\n\n" if prefix_note else "") + dynamic_summary

        return {
            "bot_message": final_msg,
            "session": session,
            "completed": True,
            "recommendation": result,
            "thinking": thinking
        }

    def generate_consultation_report(self, session: Dict[str, Any], recommendation: Dict[str, Any]) -> str:
        """
        Generates a clean, professional medical consultation & prescription report.
        """
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        age = session.get("age", "N/A")
        gender = session.get("gender", "N/A")
        severity_map = {1: "Mild (Level 1)", 2: "Moderate (Level 2)", 3: "Serious (Level 3)"}
        sev_str = severity_map.get(session.get("severity"), "N/A")
        history = session.get("history") or "None reported"
        symptoms = session.get("symptoms", [])

        diseases = recommendation.get("predicted_diseases", [])
        medicines = recommendation.get("recommended_medicines", [])
        primary_disease = diseases[0]["disease"] if diseases else "Undetermined"
        dis_info = self.recommender.disease_details.get(primary_disease, {})

        lines = []
        lines.append("================================================================================")
        lines.append("                   MEDIREC AI ASSISTANT - MEDICAL REPORT                       ")
        lines.append("================================================================================")
        lines.append(f" Report Date & Time  : {now_str}")
        lines.append(f" Course Reference    : CS619 Final Year Project (Virtual University of Pakistan)")
        lines.append(f" Student Developer   : M. Ali Sanwal (BC240440384)")
        lines.append(f" Supervisor          : Dr. Mushtaq Hussain")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(" PATIENT PROFILE & CLINICAL PRESENTATION")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f" Age                 : {age} years")
        lines.append(f" Gender              : {gender}")
        lines.append(f" Symptom Severity    : {sev_str}")
        lines.append(f" Reported Symptoms   : {', '.join(symptoms)}")
        lines.append(f" Medical History     : {history}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(" AI DIAGNOSTIC EVALUATION (Hugging Face BioBERT / ClinicalBERT Engine)")
        lines.append("--------------------------------------------------------------------------------")
        for i, d in enumerate(diseases[:3], 1):
            lines.append(f" [{i}] {d['disease']:<30} AI Confidence: {d['confidence']:.1f}%")
        lines.append(f"\n Primary Condition Summary:")
        lines.append(f" {dis_info.get('description', 'N/A')}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(" RECOMMENDED PRESCRIPTION & PHARMACOLOGICAL PROFILE")
        lines.append("--------------------------------------------------------------------------------")
        for i, med in enumerate(medicines, 1):
            mname = med.get("medicine", "N/A")
            info = med.get("info", {})
            warn = med.get("warning")
            lines.append(f"\n Medication #{i}: {mname}")
            lines.append(f"   • Generic Name     : {info.get('generic_name', 'N/A')}")
            lines.append(f"   • Drug Class       : {info.get('drug_class', 'N/A')}")
            lines.append(f"   • Indication       : Treatment for {primary_disease}")
            lines.append(f"   • Dosage           : {info.get('dosage', 'N/A')}")
            lines.append(f"   • Category         : {info.get('category', 'N/A')}")
            if info.get("side_effects"):
                lines.append(f"   • Side Effects     : {', '.join(info['side_effects'])}")
            if info.get("contraindications"):
                lines.append(f"   • Contraindications: {', '.join(info['contraindications'])}")
            if warn:
                lines.append(f"   ⚠️ CLINICAL ALERT: History matched contraindication '{warn}'!")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(" CLINICAL PRECAUTIONS & LIFESTYLE GUIDANCE")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(" Precautions:")
        for p in dis_info.get("precautions", ["Rest and monitor symptoms"]):
            lines.append(f"   • {p}")
        lines.append("\n Dietary Recommendations:")
        for dt in dis_info.get("diet", ["Balanced diet"]):
            lines.append(f"   • {dt}")
        lines.append("\n Physical Exercise & Activity:")
        for w in dis_info.get("workout", ["Adequate rest"]):
            lines.append(f"   • {w}")
        lines.append("================================================================================")
        lines.append(" DISCLAIMER: AI-generated recommendation for educational & viva evaluation only.")
        lines.append(" Always consult a licensed medical professional before administering medication.")
        lines.append("================================================================================")

        return "\n".join(lines)

    def generate_pdf_report(self, session: Dict[str, Any], recommendation: Dict[str, Any]) -> bytes:
        """
        Generates a sleek, professional, non-editable PDF Medical Consultation & Prescription Report.
        """
        from src.pdf_generator import generate_pdf_report as make_pdf
        return make_pdf(session, recommendation, self.recommender.disease_details)
