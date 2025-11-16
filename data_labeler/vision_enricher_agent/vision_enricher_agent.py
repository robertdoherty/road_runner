# data_labeler_agent/vision_enricher_agent.py
"""
Vision Enricher Agent: Uses a multimodal LLM to extract supplemental details
from post images (brand, model text, family slug, visible symptoms, etc.).

Complementary to solution_labeler_agent.py. Keeps changes additive by
attaching results under the "vision" key per post without overwriting
existing text-only labels.
"""

import os
import json
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate


def _build_llm():
    """
    Reuse the primary LLM builder. Falls back to local import when run directly.
    """
    try:
        from .data_labeler_chains import build_llm
    except ImportError:
        from data_labeler_chains import build_llm
    return build_llm()


VISION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an HVAC vision assistant. You will receive a short post context and a
list of image URLs. Use ONLY these inputs. Be conservative; return null when
unsure. Do not invent facts.

Tasks (if evidence is present):
- Extract brand (manufacturer), exact model_text (as printed), and model_family_slug
  (normalized slug) only if confident (e.g., carrier.48tc.2015_2022).
- Infer system_type (canonical list) and subtype_form_factor from visual cues (nameplate, unit form).
- For split systems, read any indoor_model_id and outdoor_model_id if visible.
- Identify visible_symptoms (e.g., burnt board, ice on coil, oil stain/leak, broken belt, tripped safety).

Output (STRICT JSON; use nulls when unsure; include confidences 0..1):
{
  "brand": "string|null",
  "brand_confidence": 0.0,
  "model_text": "string|null",
  "model_text_confidence": 0.0,
  "model_family_slug": "string|null",
  "model_family_slug_confidence": 0.0,
  # "system_type" options should be dynamically loaded for consistency.
  # Example (runtime value): "system_type": "<|>".join(SYSTEM_TYPE_CHOICES) + "|null",
  "system_type": "{system_type_choices}|null",
  "system_type_confidence": 0.0,
  "subtype_form_factor": "string|null",
  "subtype_form_factor_confidence": 0.0,
  "indoor_model_id": "string|null",
  "indoor_model_id_confidence": 0.0,
  "outdoor_model_id": "string|null",
  "outdoor_model_id_confidence": 0.0,
  "visible_symptoms": ["string"],
  "visible_symptoms_confidence": 0.0,
  "evidence_notes": "short string with what you saw (optional)"
}

Validation:
- Return ONLY valid JSON (no markdown). If unclear, keep fields null and lower confidence.
""",
    ),
    (
        "human",
        """
POST CONTEXT
post_id: {post_id}
title: {title}
problem_diagnosis_hint: {problem_diagnosis}

IMAGE URLS (review up to {max_images}):
{image_url_lines}

Return the STRICT JSON specified above.
""",
    ),
])


def safe_parse_json(text_or_obj: Any) -> Dict[str, Any]:
    """
    Parse model output as JSON. Accept dict or string; attempt to recover JSON from text.
    """
    if isinstance(text_or_obj, dict):
        return text_or_obj
    text = getattr(text_or_obj, "content", None)
    if not isinstance(text, str):
        text = str(text_or_obj)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return {"raw": text}


def _problem_diagnosis_from_labels(labels: Dict[str, Any]) -> str:
    system_info = labels.get("system_info", {}) if isinstance(labels, dict) else {}
    error_report = labels.get("error_report", {}) if isinstance(labels, dict) else {}
    system_type = system_info.get("system_type")
    symptoms = error_report.get("symptoms") or []
    parts: List[str] = []
    if system_type:
        parts.append(f"system_type={system_type}")
    if symptoms:
        parts.append("symptoms=[" + ", ".join(str(s) for s in symptoms if s) + "]")
    return "; ".join(parts) if parts else "unknown"


def run_vision_enrichment(
    enriched: Dict[str, Dict[str, Any]],
    max_images: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    For each BREAK post with image_urls, call the vision LLM with compact context
    and up to max_images URLs. Attach results under rec["vision"].
    """
    llm = _build_llm()
    chain = VISION_PROMPT | llm

    for post_id, rec in enriched.items():
        try:
            post = rec.get("post") or {}
            labels = rec.get("labels") or {}
            image_urls: List[str] = list(post.get("image_urls") or [])
            if not image_urls:
                continue

            # Select up to max_images
            selected = image_urls[: max(0, max_images)]
            if not selected:
                continue

            title = post.get("title") or ""
            problem_diag = _problem_diagnosis_from_labels(labels)

            image_url_lines = "\n".join(f"- {u}" for u in selected)
            inp = {
                "post_id": post_id,
                "title": title,
                "problem_diagnosis": problem_diag,
                "max_images": str(max_images),
                "image_url_lines": image_url_lines,
            }

            out = chain.invoke(inp)
            parsed = safe_parse_json(out)

            rec["vision"] = {
                "output": parsed,
                "input": {
                    "image_count": len(selected),
                    "image_urls": selected,
                    "title": title,
                    "problem_diagnosis": problem_diag,
                },
            }
        except Exception as e:
            rec["vision"] = {
                "error": str(e),
                "output": None,
            }

    return enriched


def write_json(data: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def process_vision_enrichment(
    raw_file: str,
    labels_file: str,
    out_file: str,
    max_images: int = 3,
) -> str:
    """
    Orchestrate: reuse existing enrich logic, then add vision enrichment, write JSON.
    """
    try:
        from .solution_labeler_agent import (
            load_break_labels,
            build_posts_index,
            enrich_breaks_with_posts,
        )
    except ImportError:
        from solution_labeler_agent import (
            load_break_labels,
            build_posts_index,
            enrich_breaks_with_posts,
        )

    breaks = load_break_labels(labels_file)
    if not breaks:
        return write_json({}, out_file)

    posts_idx = build_posts_index(raw_file, allowed_ids=set(breaks.keys()))
    enriched = enrich_breaks_with_posts(breaks, posts_idx)
    enriched = run_vision_enrichment(enriched, max_images=max_images)
    return write_json(enriched, out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision Enricher - Extract details from images for BREAK posts")
    parser.add_argument("--raw", "-r", required=True, help="Path to reddit_research_data_*.json")
    parser.add_argument("--labels", "-l", required=True, help="Path to labeled_posts_*.json")
    parser.add_argument("--output", "-o", required=True, help="Path for output vision_enriched_*.json")
    parser.add_argument("--max-images", type=int, default=3, help="Max images per post to include (default: 3)")

    args = parser.parse_args()

    print(f"🔎 Vision enriching BREAK posts with up to {args.max_images} images per post...")
    out = process_vision_enrichment(
        raw_file=args.raw,
        labels_file=args.labels,
        out_file=args.output,
        max_images=args.max_images,
    )
    print(f"✅ Vision enrichment written to: {out}")


