# data_labeler_agent/solution_labeler_agent/chains.py
"""
Minimal diagnostic labeler chain:
- Loads diagnostics (ontology), rules, and golden examples
- Builds a constrained prompt to assign up to 2 diagnostic labels
- Returns JSON with label(s), confidence(s), and rationale
"""

import os
import sys
import json
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_diagnostics_ontology() -> Dict[str, Any]:
    """Load diagnostics ontology from path specified in config."""
    root = _repo_root()
    sys.path.insert(0, root)
    try:
        from config import DIAGNOSTICS_ONTOLOGY_PATH
        ontology_path = os.path.join(root, DIAGNOSTICS_ONTOLOGY_PATH)
        return _load_json(ontology_path)
    except Exception as e:
        # Fallback to hardcoded path if config is not available
        fallback_path = os.path.join(root, "data_labeler", "rule_labeler", "meta", "diagnostics_v1.json")
        return _load_json(fallback_path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_golden_examples(path: str, max_per_label: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    try:
        data = _load_json(path)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, examples in data.items():
            if isinstance(examples, list):
                out[label] = examples[:max_per_label]
        return out
    except Exception:
        return {}


def _build_llm() -> ChatGoogleGenerativeAI:
    # Resolve API key similar to existing chain setup
    try:
        from local_secrets import GEMINI_API_KEY  # type: ignore
    except Exception:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "local_secrets", os.path.join(_repo_root(), "local_secrets.py")
            )
            local_secrets = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(local_secrets)  # type: ignore
            GEMINI_API_KEY = local_secrets.GEMINI_API_KEY  # type: ignore
        except Exception:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in local_secrets.py or environment variables")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
    )


DIAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an HVAC diagnostic classifier.

Task: Assign 0–2 PARENT diagnostic labels from the provided ontology. If unsure, output exactly dx.other_or_unclear. Be precise and conservative.

Provide:
- Allowed labels (with descriptions)
- Normalized symptoms text + raw title/body
- Equipment fields (family, subtype, brand)
- Few-shot examples by label (from real cases)

Instructions:
- Choose 0–2 labels ONLY from the allowed list.
- If unsure, return dx.other_or_unclear as the single label.
- Do NOT invent content; use only provided text.

Deterministic confidence scoring rubric (confidence method = deterministic_v1):
1. Identify distinct pieces of evidence that directly support each candidate label. Count them as ``evidence_count``.
2. Set ``base_score`` as the final confidence. Use these explicit rules:
   - Evidence definition: concrete, text-backed facts (direct quotes or specific measurements). Hearsay/speculation is not evidence.
   - Baseline by evidence count (conservative): 0 → 0.00, 1 → 0.25, 2-3 → 0.60, ≥4 → 1.00.
   - Within-bucket selection (bias conservative):
     - Default to the lower half of the bucket unless evidence is strong and precise.
     - No contradictions and evidence is strong/precise (e.g., quoted measurements, model-specific checks) → use the top of the bucket.
     - Minor uncertainty or vague phrasing → choose a lower value within the bucket (e.g., ~0.20, ~0.45, ~0.70).
     - Clear contradictions or mixed signals → choose the bottom of the bucket or the next lower bucket’s top.
   - Downshifts/penalties:
     - If symptoms are generic and could fit multiple labels, downshift one bucket.
   - Special caps:
     - If the only plausible label is dx.other_or_unclear, cap ``base_score`` at 0.30.
     - If all evidence is speculative (no concrete facts), treat as 0 evidence.
3. Report ``confidence`` = clamp(base_score, 0.0, 1.0) rounded to two decimals.

OUTPUT (STRICT JSON, no markdown):
{{
  "predictions": [
    {{
      "label_id": "string",
      "confidence": 0.0,
      "confidence_breakdown": {{
        "method": "deterministic_v1",
        "evidence_count": 0,
        "base_score": 0.0
      }},
      "rationale": "string"
    }}
  ]
}}

Validation:
- Labels MUST be in the allowed set; otherwise use dx.other_or_unclear.
- 0–2 predictions only; sort by confidence descending.
- Return VALID JSON only (no markdown).
"""
    ),
    (
        "human",
        """
ALLOWED LABELS (you MUST choose only from this list; if none fit, use dx.other_or_unclear)
{labels_block}

FEW-SHOT EXAMPLES (per label, up to 3)
{examples_block}

INPUT
post_id: {post_id}
title: {title}
body: {body}
equipment: {equip}
normalized_symptoms: {x_symptoms}
"""
    ),
])


def build_diagnostic_labeler_chain() -> RunnableSequence:
    llm = _build_llm()
    return DIAG_PROMPT | llm


def _load_label_details_from_golden() -> Dict[str, Dict[str, str]]:
    """Fallback helper to pull descriptions/effects from the golden set."""
    root = _repo_root()
    sys.path.insert(0, root)
    from config import GOLDEN_DIAGNOSTIC_CHART_PATH

    golden_path = os.path.join(root, GOLDEN_DIAGNOSTIC_CHART_PATH)
    golden_data = _load_json(golden_path)
    diagnostics = golden_data.get("diagnostics", []) or []

    details: Dict[str, Dict[str, str]] = {}
    for diag in diagnostics:
        diag_id = diag.get("diagnostic_id")
        if not diag_id:
            continue
        details[diag_id] = {
            "description": diag.get("description")
            or diag.get("specific_diagnostic_name")
            or "",
            "typical_effect_on_operation": diag.get("typical_effect_on_operation", ""),
        }
    return details


def render_allowed_labels(ontology: Dict[str, Any]) -> str:
    """Render allowed labels with descriptions and typical effects."""
    labels = ontology.get("labels", [])
    if not labels:
        raise ValueError("No labels found in ontology")

    details = ontology.get("label_details") or {}
    # Fallback to golden set if necessary
    if not details or any(lid not in details for lid in labels):
        details = _load_label_details_from_golden()

    lines = []
    for lid in labels:
        info = details.get(lid, {})
        description = info.get("description") or ""
        typical_effect = info.get("typical_effect_on_operation") or ""
        if not description and not typical_effect:
            raise ValueError(f"No description/effect found for label '{lid}'")
        text = f"- {lid}: {description}".strip()
        if typical_effect:
            text = f"{text} | Typical effect: {typical_effect}"
        lines.append(text)

    return "\n".join(lines)


def render_examples_block(gold: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    for label, examples in gold.items():
        lines.append(f"[{label}]")
        for ex in examples:
            text = ex.get("text", "")
            equip = ex.get("equip", {})
            lines.append(f"  - text: {text}\n    equip: {equip}")
    return "\n".join(lines)



