# data_labeler_agent/solution_labeler_agent/agent.py
"""
Minimal diagnostic agent orchestrator.

Loads ontology, rules, and golden examples. Builds input from a post record
and calls a constrained LLM chain to predict up to 2 diagnostic labels.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .diagnostic_schema import DiagnosticInput, DiagnosticOutput, enforce_allowed_predictions
from .diagnostic_chains import (
    build_diagnostic_labeler_chain,
    render_allowed_labels,
    render_examples_block,
    _load_json,
    _load_golden_examples,
)

try:
    from config import DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY
except Exception:
    DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY = 3


logger = logging.getLogger(__name__)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _normalize_symptoms(symptoms: list[str], title: str, body: str) -> str:
    # Simple concatenation; reuse existing make_error_prediction normalize if desired later
    clean = [s.strip() for s in (symptoms or []) if isinstance(s, str) and s.strip()]
    if clean:
        return "; ".join(clean)
    text = f"{title} {body}".strip()
    return " ".join(text.split())


def build_llm_payload(
    post: DiagnosticInput,
    ontology: Dict[str, Any],
    gold_examples: Dict[str, Any],
    confidence_max: float = 1.0,
) -> Dict[str, Any]:
    labels_block = render_allowed_labels(ontology)
    examples_block = render_examples_block(gold_examples)
    x_symptoms = _normalize_symptoms(post.get("symptoms", []), post.get("title", ""), post.get("body", ""))

    return {
        "labels_block": labels_block,
        "examples_block": examples_block,
        "post_id": post.get("post_id", ""),
        "title": post.get("title", ""),
        "body": post.get("body", ""),
        "equip": post.get("equip", {}),
        "x_symptoms": x_symptoms,
        "confidence_max": confidence_max,
    }


def predict_diagnostics_batch(
    posts: Sequence[DiagnosticInput],
    max_concurrency: Optional[int] = None,
    confidence_max: Optional[float] = None,
) -> List[DiagnosticOutput]:
    """Run the diagnostic labeler for multiple posts with optional concurrency."""
    if not posts:
        return []

    root = _repo_root()
    
    # Load paths from config
    sys.path.insert(0, root)
    try:
        from config import DIAGNOSTICS_ONTOLOGY_PATH
        ontology_path = os.path.join(root, DIAGNOSTICS_ONTOLOGY_PATH)
    except Exception:
        # Fallback to hardcoded path
        ontology_path = os.path.join(root, "data_labeler", "rule_labeler", "meta", "diagnostics_v1.json")
    
    gold_path = os.path.join(root, "data_labeler", "rule_labeler", "gold", "golden_examples.json")

    ontology = _load_json(ontology_path)
    gold = _load_golden_examples(gold_path, max_per_label=3)

    chain = build_diagnostic_labeler_chain()

    payloads: List[Dict[str, Any]] = [
        build_llm_payload(
            post,
            ontology,
            gold,
            confidence_max=(confidence_max if isinstance(confidence_max, (int, float)) else 1.0),
        )
        for post in posts
    ]

    configured_concurrency = (
        max_concurrency
        if isinstance(max_concurrency, int) and max_concurrency > 0
        else DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY
    )

    logger.info(
        "Invoking diagnostic labeler chain for %d posts (max_concurrency=%d)",
        len(payloads),
        configured_concurrency,
    )

    try:
        outputs = chain.batch(
            payloads,
            config={"max_concurrency": configured_concurrency},
        )
    except Exception as batch_exc:
        logger.exception(
            "Batch execution failed (max_concurrency=%d): %s; falling back to sequential processing.",
            configured_concurrency,
            batch_exc,
        )
        outputs = []
        for idx, payload in enumerate(payloads, 1):
            try:
                outputs.append(chain.invoke(payload))
            except Exception as invoke_exc:
                logger.exception(
                    "Sequential diagnostic invoke failed for post %d/%d: %s",
                    idx,
                    len(payloads),
                    invoke_exc,
                )
                outputs.append(invoke_exc)
        logger.info("Sequential fallback complete (%d posts)", len(outputs))
    else:
        logger.info("Batch execution complete.")

    if len(outputs) != len(payloads):
        logger.warning(
            "Output count (%d) does not match payload count (%d).",
            len(outputs),
            len(payloads),
        )

    allowed = set(ontology.get("labels", []))
    results: List[DiagnosticOutput] = []

    for idx, post in enumerate(posts):
        output = (
            outputs[idx]
            if idx < len(outputs)
            else Exception("missing output from batch execution")
        )
        if isinstance(output, Exception):
            logger.exception(
                "Diagnostic labeler failed for post_id=%s",
                post.get("post_id"),
            )
            raise output

        text = output.content if hasattr(output, "content") else str(output)

        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except Exception:
                    parsed = {}
            else:
                parsed = {}

        preds = parsed.get("predictions", []) if isinstance(parsed, dict) else []

        results.append(
            {
                "predictions": enforce_allowed_predictions(preds, allowed, max_labels=2),
                "ontology": "diagnostics_v1",
                "provenance": "llm_v1",
            }
        )

    return results


def predict_diagnostics(
    post: DiagnosticInput,
    max_concurrency: Optional[int] = None,
    confidence_max: Optional[float] = None,
) -> DiagnosticOutput:
    """Run the diagnostic labeler for a single post using the batch implementation."""
    results = predict_diagnostics_batch(
        [post],
        max_concurrency=max_concurrency,
        confidence_max=confidence_max,
    )
    if not results:
        raise ValueError("Diagnostic agent returned no predictions")
    return results[0]


__all__ = [
    "predict_diagnostics",
    "predict_diagnostics_batch",
]
