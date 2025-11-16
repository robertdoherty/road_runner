# data_labeler_agent/solution_labeler_agent.py
"""
Solution Labeler Agent: Processes BREAK-labeled posts to find solutions from comments.
Minimal, efficient pipeline with per-post error isolation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    # Threshold for accepting comment-derived fields and concurrency defaults
    from config import (
        COMMENT_ENRICHMENT_MIN_CONFIDENCE,
        DEFAULT_SOLUTION_MAX_CONCURRENCY,
    )
except Exception:
    COMMENT_ENRICHMENT_MIN_CONFIDENCE = 0.6
    DEFAULT_SOLUTION_MAX_CONCURRENCY = 3

logger = logging.getLogger(__name__)


def load_break_labels(labels_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load labels JSON; return dict {post_id: label_json} for break_label == "BREAK".
    
    Args:
        labels_file: Path to labeled_posts_*.json file
        
    Returns:
        Dictionary mapping post_id to full label JSON for BREAK posts only
    """
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_doc = json.load(f)
    
    break_labels: Dict[str, Dict[str, Any]] = {}
    
    for result in labels_doc.get("results", []):
        if not isinstance(result, dict):
            continue
        
        # Get post ID from the 'id' field in labels
        post_id = result.get("id")
        if not post_id:
            continue
        
        # Only include BREAK posts (nested under error_report)
        error_report = result.get("error_report") or {}
        if error_report.get("break_label") == "BREAK":
            break_labels[post_id] = result
    
    return break_labels


def build_posts_index(raw_file: str, allowed_ids: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load raw JSON; return dict {post_id: post} for posts where post_id (or fallback id) 
    is in allowed_ids (if provided).
    
    Args:
        raw_file: Path to reddit_research_data_*.json file
        allowed_ids: Optional set of post IDs to include (for efficiency)
        
    Returns:
        Dictionary mapping post_id to full post object with all metadata and comments
    """
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    posts_index: Dict[str, Dict[str, Any]] = {}
    
    # Navigate through subreddits structure
    subreddits = raw_data.get("subreddits", {})
    for subreddit_name, subreddit_data in subreddits.items():
        for post in subreddit_data.get("posts", []):
            # Try 'post_id' first, fallback to 'id'
            post_id = post.get("post_id") or post.get("id")
            if not post_id:
                continue
            
            # If allowed_ids provided, only include those posts
            if allowed_ids is not None and post_id not in allowed_ids:
                continue
            
            # Store the full post object
            posts_index[post_id] = post
    
    return posts_index


def enrich_breaks_with_posts(break_labels: Dict[str, Dict[str, Any]], 
                              posts_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Join into {post_id: {"post_id": ..., "labels": <labels>, "post": <full_post>}}. 
    Skip missing posts.
    
    Args:
        break_labels: Dictionary of BREAK labels by post_id
        posts_index: Dictionary of full post objects by post_id
        
    Returns:
        Enriched dictionary with both labels and full post data
    """
    enriched: Dict[str, Dict[str, Any]] = {}
    
    for post_id, labels in break_labels.items():
        # Look up the full post
        post = posts_index.get(post_id)
        
        # Skip if post not found in raw data
        if not post:
            continue
        
        # Build enriched structure
        enriched[post_id] = {
            "post_id": post_id,
            "labels": labels,
            "post": post  # Full, unmodified post with all metadata and comments
        }
    
    return enriched


def _ensure_nested(labels: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure nested structures exist with default shapes."""
    if not isinstance(labels, dict):
        labels = {}
    labels.setdefault("error_report", {})
    labels.setdefault("system_info", {})
    labels.setdefault("_evidence", {})
    labels.setdefault("_provenance", {})
    labels.setdefault("_conflicts", [])
    # Default arrays
    labels["error_report"].setdefault("symptoms", [])
    labels["error_report"].setdefault("error_codes", [])
    return labels


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _append_evidence(labels: Dict[str, Any], field: str, ids: List[str]):
    ev = labels.setdefault("_evidence", {})
    cur = list(ev.get(field, []))
    if ids:
        cur.extend(ids)
        ev[field] = _dedupe_preserve_order(cur)


def _record_provenance(labels: Dict[str, Any], field: str, provenance: Optional[str]):
    if not provenance:
        return
    prov = labels.setdefault("_provenance", {})
    prov[field] = provenance


def _record_conflict(labels: Dict[str, Any], field: str, base_val: Any, new_val: Any):
    labels.setdefault("_conflicts", []).append({
        "field": field,
        "base": base_val,
        "comments": new_val,
    })


def _merge_error_report_delta(labels: Dict[str, Any], delta: Dict[str, Any]) -> None:
    er = labels.setdefault("error_report", {})
    # Only OP-derived fields should be accepted; enforced by caller via provenance check
    evidence_by = delta.get("evidence_refs_by_field") or {}
    prov_by = delta.get("provenance_by_field") or {}
    conf_by = delta.get("field_confidence_by_field") or {}

    for field in ("symptoms", "error_codes"):
        items = delta.get(field) or []
        provenance = prov_by.get(field)
        conf = float(conf_by.get(field, 0.0) or 0.0)
        if not items:
            continue
        if provenance not in ("op_comment", "op_edit"):
            continue
        if conf < COMMENT_ENRICHMENT_MIN_CONFIDENCE:
            continue
        base_items = list(er.get(field) or [])
        base_items.extend(items)
        er[field] = _dedupe_preserve_order(base_items)
        _append_evidence(labels, field, evidence_by.get(field) or [])
        _record_provenance(labels, field, provenance)


def _merge_system_info_delta(labels: Dict[str, Any], delta: Dict[str, Any]) -> None:
    si = labels.setdefault("system_info", {})
    evidence_by = delta.get("evidence_refs_by_field") or {}
    prov_by = delta.get("provenance_by_field") or {}
    conf_by = delta.get("field_confidence_by_field") or {}

    string_fields = [
        "system_type",
        "asset_subtype",
        "brand",
        "model_text",
        "model_family_id",
        "indoor_model_id",
        "outdoor_model_id",
    ]

    for field in string_fields:
        new_val = delta.get(field)
        if not new_val:
            continue
        provenance = prov_by.get(field)
        conf = float(conf_by.get(field, 0.0) or 0.0)
        if provenance not in ("op_comment", "op_edit"):
            continue
        if conf < COMMENT_ENRICHMENT_MIN_CONFIDENCE:
            continue
        base_val = si.get(field, "")
        if not base_val:
            si[field] = new_val
            _record_provenance(labels, field, provenance)
            _append_evidence(labels, field, evidence_by.get(field) or [])
        elif base_val != new_val:
            # OP-only override allowed; record conflict then override
            _record_conflict(labels, field, base_val, new_val)
            si[field] = new_val
            _record_provenance(labels, field, provenance)
            _append_evidence(labels, field, evidence_by.get(field) or [])

    # Model resolution confidence: take max
    try:
        new_conf = float(delta.get("model_resolution_confidence", 0.0) or 0.0)
    except Exception:
        new_conf = 0.0
    try:
        base_conf = float(si.get("model_resolution_confidence", 0.0) or 0.0)
    except Exception:
        base_conf = 0.0
    si["model_resolution_confidence"] = max(base_conf, new_conf)


def _merge_enrichment_into_labels(labels: Dict[str, Any], enrichment: Dict[str, Any]) -> Dict[str, Any]:
    labels = _ensure_nested(labels)
    if not isinstance(enrichment, dict):
        return labels
    er_delta = enrichment.get("error_report_delta") or {}
    si_delta = enrichment.get("system_info_delta") or {}
    if er_delta:
        _merge_error_report_delta(labels, er_delta)
    if si_delta:
        _merge_system_info_delta(labels, si_delta)
    return labels


def run_solutions_on_enriched(
    enriched: Dict[str, Dict[str, Any]],
    max_concurrency: Optional[int] = None,
    output_dir: str = "output",
) -> Dict[str, Dict[str, Any]]:
    """
    Iterate through all BREAK posts and call build_solution_labeler_chain for each.
    Pass the full enriched data (no alterations). Append LLM output to each record.
    
    Args:
        enriched: Dictionary of enriched BREAK posts with labels and full post data
        max_concurrency: Optional override for concurrent LLM calls.
        
    Returns:
        Enriched dictionary with solution appended to each post
    """
    logger.info("=== Solution Labeler: Starting run_solutions_on_enriched ===")
    
    # Import the chain builder
    try:
        from .solution_labeler_chains import build_solution_labeler_chain
    except ImportError:
        from solution_labeler_chains import build_solution_labeler_chain
    
    chain = build_solution_labeler_chain()
    logger.debug("Solution labeler chain built successfully")

    prepared_payloads: List[Dict[str, Any]] = []
    prepared_records: List[Tuple[str, Dict[str, Any]]] = []

    # Pre-build payloads so we can batch execute while maintaining per-post isolation
    for post_id, rec in enriched.items():
        try:
            post = rec.get("post", {})
            labels = rec.get("labels", {})

            # Extract what we need for the LLM
            title = post.get("title", "")
            user_id = post.get("author", "")
            comments = post.get("comments", [])

            # Build problem diagnosis from labels (nested)
            labels = _ensure_nested(labels)
            rec["labels"] = labels
            symptoms = labels.get("error_report", {}).get("symptoms", [])
            system_type = labels.get("system_info", {}).get("system_type")
            problem_diagnosis = (
                f"system_type={system_type}; symptoms={symptoms}"
                if system_type or symptoms
                else "unknown"
            )

            # Convert comments to JSON string for the LLM
            comments_json = json.dumps(comments, ensure_ascii=False)

            # Prepare input for build_solution_labeler_chain
            chain_input = {
                "post_id": post_id,
                "title": title,
                "user_id": user_id,
                "problem_diagnosis": problem_diagnosis,
                "comments_json": comments_json,
            }

            prepared_payloads.append(chain_input)
            prepared_records.append((post_id, rec))
        except Exception as e:
            rec["solution_report"] = {
                "summary": "No clear solution.",
                "error": str(e),
                "confidence": 0.0,
            }

    if not prepared_payloads:
        logger.info("No BREAK posts ready for solution labeling batch execution.")
        return enriched

    configured_concurrency = (
        max_concurrency
        if isinstance(max_concurrency, int) and max_concurrency > 0
        else DEFAULT_SOLUTION_MAX_CONCURRENCY
    )

    logger.info(
        "Invoking solution labeler chain for %d posts (max_concurrency=%d)",
        len(prepared_payloads),
        configured_concurrency,
    )
    logger.debug(f"Post IDs being processed: {[p['post_id'] for p in prepared_payloads]}")

    try:
        import time
        batch_start = time.time()
        logger.debug(f"Starting batch execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        outputs = chain.batch(
            prepared_payloads,
            config={"max_concurrency": configured_concurrency},
        )
        
        batch_duration = time.time() - batch_start
        logger.info(f"Batch execution completed in {batch_duration:.2f}s")
    except Exception as batch_exc:
        logger.exception(
            "Batch execution failed (max_concurrency=%d): %s; falling back to sequential processing.",
            configured_concurrency,
            batch_exc,
        )
        outputs = []
        seq_start = time.time()
        for idx, payload in enumerate(prepared_payloads):
            post_id = payload.get('post_id', f'unknown_{idx}')
            try:
                logger.debug(f"Sequential invoke starting for post {post_id}")
                invoke_start = time.time()
                output = chain.invoke(payload)
                invoke_duration = time.time() - invoke_start
                logger.debug(f"Sequential invoke completed for post {post_id} in {invoke_duration:.2f}s")
                outputs.append(output)
            except Exception as invoke_exc:
                logger.error(f"Sequential invoke failed for post {post_id}: {invoke_exc}")
                outputs.append(invoke_exc)
        seq_duration = time.time() - seq_start
        logger.info(
            "Sequential fallback complete (%d posts) in {seq_duration:.2f}s",
            len(outputs),
        )
    else:
        logger.info("Batch execution complete.")

    if len(outputs) != len(prepared_records):
        logger.warning(
            "Output count (%d) does not match prepared record count (%d).",
            len(outputs),
            len(prepared_records),
        )
    # Merge outputs into enriched records
    for idx, (post_id, rec) in enumerate(prepared_records):
        try:
            output = outputs[idx] if idx < len(outputs) else Exception(
                "missing output from batch execution"
            )
            if isinstance(output, Exception):
                raise output

            # Parse the output (handle if it's a LangChain message object)
            if hasattr(output, "content"):
                solution_text = output.content
            else:
                solution_text = str(output)

            # Try to parse as JSON
            try:
                solution = json.loads(solution_text)
            except Exception:
                # If parsing fails, extract JSON from text
                start = solution_text.find("{")
                end = solution_text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        solution = json.loads(solution_text[start : end + 1])
                    except Exception:
                        solution = {"raw": solution_text}
                else:
                    solution = {"raw": solution_text}

            # Extract solution_report and optional enrichment deltas
            solution_report = solution.get("solution_report") if isinstance(solution, dict) else None
            if solution_report is None:
                # Back-compat or raw: store under solution_report as best-effort
                solution_report = {"summary": str(solution)}

            # Merge enrichment deltas into labels with OP-only policy and threshold
            enrichment_block = solution.get("enrichment") if isinstance(solution, dict) else None
            if isinstance(enrichment_block, dict):
                labels = rec.get("labels", {})
                labels = _merge_enrichment_into_labels(labels, enrichment_block)
                rec["labels"] = labels

            # Append solution report to the current record
            rec["solution_report"] = solution_report

        except Exception as e:
            # Per-post error isolation
            rec["solution_report"] = {
                "summary": "No clear solution.",
                "error": str(e),
                "confidence": 0.0,
            }

    return enriched


def write_json(data: Dict[str, Any], out_path: str) -> str:
    """
    Write JSON to disk. Keep simple.
    
    Args:
        data: Dictionary to write
        out_path: Output file path
        
    Returns:
        Path to written file
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return out_path


def process_breaks_to_solutions(
    raw_file: str,
    labels_file: str,
    out_file: str,
    max_concurrency: Optional[int] = None,
    output_dir: str = "output",
) -> str:
    """
    Orchestrate the full pipeline.
    
    Args:
        raw_file: Path to reddit_research_data_*.json
        labels_file: Path to labeled_posts_*.json
        out_file: Path for output solutions_*.json
        max_concurrency: Optional override for concurrent LLM calls.
        
    Returns:
        Path to output file
    """
    # Load BREAK labels
    break_labels = load_break_labels(labels_file)
    
    # Early return if no BREAK posts
    if not break_labels:
        return write_json({}, out_file)
    
    # Build posts index (only for BREAK post IDs)
    posts_index = build_posts_index(raw_file, allowed_ids=set(break_labels.keys()))
    
    # Enrich: join labels + full posts
    enriched = enrich_breaks_with_posts(break_labels, posts_index)
    
    # Run solution finding on all enriched posts
    solved = run_solutions_on_enriched(
        enriched,
        max_concurrency=max_concurrency,
        output_dir=output_dir,
    )
    
    # Write to disk
    return write_json(solved, out_file)


# CLI interface when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solution Labeler - Find solutions from comments for BREAK posts")
    parser.add_argument("--raw", "-r", required=True, help="Path to reddit_research_data_*.json")
    parser.add_argument("--labels", "-l", required=True, help="Path to labeled_posts_*.json")
    parser.add_argument("--output", "-o", required=True, help="Path for output solutions_*.json")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Optional override for concurrent LLM calls",
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Loading BREAK labels from: {args.labels}")
    print(f"📋 Loading raw posts from: {args.raw}")
    print(f"🤖 Processing with LLM...")
    
    output_file = process_breaks_to_solutions(
        raw_file=args.raw,
        labels_file=args.labels,
        out_file=args.output,
        max_concurrency=args.max_concurrency,
    )
    
    print(f"✅ Solutions written to: {output_file}")

