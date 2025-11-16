# data_labeler_agent/break_labeler_chains.py
"""
Break Labeler Chain: Classifies Reddit posts as BREAK/NON_BREAK and extracts
structured HVAC malfunction details (system_type, brand, model, symptoms, etc.).
"""

import os
import sys
import json
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Import the Pydantic schema and parser
try:
    from .break_labeler_schema import parser, BreakOutput
except ImportError:
    from break_labeler_schema import parser, BreakOutput

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_system_types() -> dict:
    """Load system_types.json generated from golden_set_v3.

    Returns a dict with:
      - system_types: list[str]
      - by_diagnostic_id: dict[str, list[str]]
    """
    path = os.path.join(
        _repo_root(),
        "data_labeler",
        "rule_labeler",
        "meta",
        "system_types.json",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback to empty structure if file not present yet
        return {"system_types": [], "by_diagnostic_id": {}}


SYSTEM_TYPES = _load_system_types()

try:
    from local_secrets import GEMINI_API_KEY
except ImportError:
    try:
        import importlib.util
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        spec = importlib.util.spec_from_file_location("local_secrets", os.path.join(repo_root, "local_secrets.py"))
        local_secrets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_secrets)
        GEMINI_API_KEY = local_secrets.GEMINI_API_KEY
    except:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in local_secrets.py or environment variables")


def build_llm() -> ChatGoogleGenerativeAI:
    """Build the Gemini 2.5 Flash model instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.25,
        google_api_key=GEMINI_API_KEY,
    )


LABELER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You label individual Reddit posts about HVAC. Decide if a post describes a concrete malfunction requiring diagnosis/repair.

Rules:
- Output STRICT JSON only (no prose).
- Prefer precision over recall. If unsure => NON_BREAK.
- Extract ONLY from the post (title/body/metadata). Don't invent.

BREAK criteria (need >=1):
- Concrete symptom/failure (no cooling/heat, trips, leak, won't start, unusual noise/odor, error code).
- Measured abnormality implying malfunction (pressures/temps/amps).

NON_BREAK examples:
- Shopping/quotes, brand opinions, install pics w/o issues, non-HVAC.

Fields available per post: title, body, score, num_comments, upvote_ratio, media(if present).

Canonical system_types (choose one per post; use "other_or_unclear" if unknown):
{system_types_list}, other_or_unclear

Glossary:
- break_label: "BREAK" or "NON_BREAK" classification
- symptoms: array of symptom strings describing the malfunction
- error_codes: controller codes/tokens (E5, 33, LPS)
- system_type: canonical HVAC system bucket (examples listed above)
- asset_subtype: physical/config form (e.g., packaged RTU, ductless wall, cassette, horizontal)
- brand: manufacturer token as written (Carrier, Trane, Goodman)
- model_text: raw manufacturer model number string(s) ONLY if clearly present (e.g., "ML193UH070XP36B-58", "RGPE-07EAMKR"). Must look like a factory model ID (letters/numbers, optional dashes/dots). 
- model_family_id: normalized brand+series (optional broad year bucket) ONLY if  confident. Format: "brand.series" or "brand.series.2015_2022" (lowercase, dot-separated), e.g., "carrier.48tc", "daikin.air_cooled_chiller.2015_2022". If you only know brand+family, use an "unknown_*" suffix (e.g., "goodman.unknown_furnace"). If not confident, use "".

Type rules (CRITICAL - must follow exactly):
- Confidences are floats 0.0-1.0 (NOT strings, NOT null).
- Arrays => [] when none (NOT null).
- Strings => "" when unknown (NOT null).
- Booleans => true/false (NOT null).
- break_label: MUST be exactly "BREAK" or "NON_BREAK"
- system_type: MUST be one of the allowed values (including "other_or_unclear")
- All confidence fields MUST be present with float values 0.0-1.0

Output (one object per input post):
{{
  "results": [
    {{
      "id": "<post id>",
      "error_report": {{
        "break_label": "BREAK" | "NON_BREAK",
        "break_confidence": 0.0,
        "symptoms": [],
        "symptoms_confidence": 0.0,
        "error_codes": [],
        "error_codes_confidence": 0.0
      }},
      "system_info": {{
        "system_type": "",
        "system_type_confidence": 0.0,
        "asset_subtype": "",
        "asset_subtype_confidence": 0.0,
        "brand": "",
        "brand_confidence": 0.0,
        "model_text": "",
        "model_text_confidence": 0.0,
        "model_family_id": "",
        "model_family_id_confidence": 0.0
      }}
    }}
  ]
}}
"""),
    ("human", "Reddit JSON data follows:\n```json\n{json_data}\n```")
])


def build_data_labeler_chain() -> RunnableSequence:
    """
    Build the BREAK/NON_BREAK labeling chain with Pydantic schema enforcement
    
    Returns:
        RunnableSequence that outputs validated BreakOutput objects
    """
    llm = build_llm()
    system_types_list = ", ".join(SYSTEM_TYPES.get("system_types", []))
    prompt = LABELER_PROMPT.partial(system_types_list=system_types_list)
    chain = prompt | llm
    return chain
