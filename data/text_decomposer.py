"""
LLM-Guided Semantic Text Decomposer for DAMR-LLM
Decomposes raw natural language text queries Q into two disentangled sub-prompts:
  1. Q_app: Appearance Sub-prompt (Clothing, colors, physical appearance, accessories)
  2. Q_mot: Motion Sub-prompt (Actions, direction, movement, interactions)
Supports deterministic POS-parsing fallback and pre-computed LLM annotations cache.
"""

import re
from typing import Dict, List, Tuple, Optional, Any

# Rule-based semantic vocabulary for English TVPR
CLOTHING_ITEMS_EN = {
    "coat", "jacket", "shirt", "t-shirt", "top", "blouse", "sweater", "hoodie", "cardigan",
    "vest", "suit", "blazer", "pants", "trousers", "jeans", "shorts", "skirt", "dress",
    "leggings", "sweatpants", "shoes", "sneakers", "boots", "sandals", "trainers", "heels",
    "backpack", "bag", "handbag", "purse", "shoulder bag", "plastic bag", "luggage",
    "suitcase", "briefcase", "umbrella", "hat", "cap", "beanie", "glasses", "sunglasses"
}

COLORS_EN = {
    "black", "white", "red", "blue", "green", "yellow", "brown", "grey", "gray",
    "dark", "light", "orange", "pink", "purple", "beige", "navy", "tan", "maroon",
    "cream", "khaki", "olive", "patterned", "striped", "plaid"
}

APPEARANCE_TRAITS_EN = {
    "man", "woman", "person", "male", "female", "guy", "lady", "boy", "girl",
    "hair", "short-hair", "long-hair", "bald", "tall", "short", "slim", "thin"
}

MOTION_VERBS_EN = {
    "walk", "walking", "walks", "walked", "run", "running", "runs", "ran",
    "jog", "jogging", "jogs", "turn", "turning", "turns", "turned",
    "carry", "carrying", "carries", "carried", "hold", "holding", "holds", "held",
    "cross", "crossing", "crosses", "crossed", "enter", "entering", "enters",
    "move", "moving", "moves", "moved", "stop", "stopping", "stops",
    "stand", "standing", "stands", "stood", "go", "going", "goes", "went",
    "approach", "approaching", "approaches", "leave", "leaving", "leaves",
    "pull", "pulling", "pulls", "push", "pushing", "ride", "riding"
}

MOTION_DIRECTIONS_EN = {
    "towards", "away", "across", "along", "into", "out", "left", "right",
    "straight", "forward", "backward", "slowly", "fast", "quickly"
}

# Rule-based vocabulary for Vietnamese
MOTION_VERBS_VI = {
    "đi", "đi_bộ", "chạy", "quay", "ngoái", "rẽ", "bước", "xách", "cầm", "mang",
    "bỏ", "đặt", "đang_đi", "đang_chạy", "bước_đi", "di_chuyển", "qua_đường",
    "tiến", "lại", "rời", "dừng", "đứng", "kéo", "đẩy"
}


class TextDecomposer:
    """
    Disentangled Semantic Concept Alignment (DSCA) Text Decomposer.
    Extracts fine-grained appearance keywords, motion keywords, and structured sub-prompts:
      1. Q_app: Appearance Sub-prompt (Clothing, colors, physical appearance, accessories)
      2. Q_mot: Motion Sub-prompt (Actions, direction, movement dynamics)
      3. Concepts: Dictionary of extracted visual and dynamic tokens for academic explainability
    """
    def __init__(self, language: str = "en", llm_cache: Optional[Dict[str, Dict[str, str]]] = None):
        self.language = language.lower()
        self.llm_cache = llm_cache or {}

    def extract_concepts(self, text: str) -> Dict[str, Any]:
        """
        Extracts structured semantic concepts from a natural language person description.
        Returns:
            Dict containing 'app_keywords', 'mot_keywords', 'q_app', 'q_mot', 'q_full'.
        """
        text_clean = text.strip()
        words = re.findall(r'\b[a-zA-Z_\-]+\b', text_clean.lower())

        app_keywords: List[str] = []
        mot_keywords: List[str] = []

        if self.language == "vi":
            q_app, q_mot = self._decompose_vi(text_clean)
            return {
                "app_keywords": app_keywords,
                "mot_keywords": mot_keywords,
                "q_app": q_app,
                "q_mot": q_mot,
                "q_full": text_clean
            }

        # English extraction
        skip_next = False
        gender_traits = []
        for i, w in enumerate(words):
            if skip_next:
                skip_next = False
                continue

            if w in APPEARANCE_TRAITS_EN:
                if w in {"man", "woman", "person", "male", "female", "guy", "lady", "boy", "girl"}:
                    if w not in gender_traits:
                        gender_traits.append(w)
                elif w not in app_keywords:
                    app_keywords.append(w)
            elif w in COLORS_EN or w in CLOTHING_ITEMS_EN:
                if w in COLORS_EN and i + 1 < len(words) and words[i + 1] in CLOTHING_ITEMS_EN:
                    pair = f"{w} {words[i + 1]}"
                    if pair not in app_keywords:
                        app_keywords.append(pair)
                    skip_next = True
                elif w not in app_keywords:
                    app_keywords.append(w)
            elif w in MOTION_VERBS_EN or w in MOTION_DIRECTIONS_EN:
                if w not in mot_keywords:
                    mot_keywords.append(w)

        subject = gender_traits[0] if gender_traits else "person"
        q_app, q_mot = self._decompose_en(text_clean, app_keywords, mot_keywords, subject=subject)

        return {
            "app_keywords": app_keywords,
            "mot_keywords": mot_keywords,
            "q_app": q_app,
            "q_mot": q_mot,
            "q_full": text_clean
        }

    def decompose(self, text: str) -> Tuple[str, str]:
        """
        Decomposes input text string into (Q_app, Q_mot).
        """
        text_clean = text.strip()
        if text_clean in self.llm_cache:
            item = self.llm_cache[text_clean]
            return item.get("Q_app", text_clean), item.get("Q_mot", text_clean)

        concepts = self.extract_concepts(text_clean)
        return concepts["q_app"], concepts["q_mot"]

    def _decompose_en(
        self,
        text: str,
        app_kws: Optional[List[str]] = None,
        mot_kws: Optional[List[str]] = None,
        subject: str = "person"
    ) -> Tuple[str, str]:
        clauses = re.split(r'[,;]|\b(?:while|and then|then)\b', text, flags=re.IGNORECASE)
        app_clauses = []
        mot_clauses = []

        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean:
                continue
            clause_lower = clause_clean.lower()
            is_motion = any(v in clause_lower for v in MOTION_VERBS_EN)
            if is_motion:
                mot_clauses.append(clause_clean)
            else:
                app_clauses.append(clause_clean)

        # Formulate concise, focused sub-prompts
        if app_kws and len(app_kws) >= 1:
            q_app = f"A {subject} wearing {', '.join(app_kws)}"
        elif app_clauses:
            q_app = f"A photo of a {subject}: {', '.join(app_clauses)}"
        else:
            q_app = f"A photo of a {subject}: {text}"

        if mot_kws and len(mot_kws) >= 1:
            q_mot = f"A video of a {subject} {', '.join(mot_kws)}"
        elif mot_clauses:
            q_mot = f"A video of a {subject} performing action: {', '.join(mot_clauses)}"
        else:
            q_mot = f"A video of a {subject} walking and moving: {text}"

        return q_app, q_mot

    def _decompose_vi(self, text: str) -> Tuple[str, str]:
        clauses = re.split(r'[,;]|\b(?:sau đó|vừa|rồi|khi)\b', text, flags=re.IGNORECASE)
        app_words = []
        mot_words = []

        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean:
                continue
            clause_lower = clause_clean.lower()
            is_motion = any(v in clause_lower for v in MOTION_VERBS_VI)
            if is_motion:
                mot_words.append(clause_clean)
            else:
                app_words.append(clause_clean)

        q_app = ", ".join(app_words) if app_words else text
        q_mot = ", ".join(mot_words) if mot_words else f"hành động và di chuyển của {text}"

        q_app = f"Hình ảnh người có trang phục: {q_app}"
        q_mot = f"Video người thực hiện hành động: {q_mot}"
        return q_app, q_mot

    def decompose_batch(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        app_list, mot_list = [], []
        for t in texts:
            q_app, q_mot = self.decompose(t)
            app_list.append(q_app)
            mot_list.append(q_mot)
        return app_list, mot_list
