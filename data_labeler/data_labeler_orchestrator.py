"""Data Labeler Agent: Orchestrates the full HVAC labeling pipeline."""

import json
import csv
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

# Ensure imports work across direct script and package contexts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
repo_root = os.path.dirname(current_dir)
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)
logger = logging.getLogger(__name__)

data_agent_dir = os.path.join(current_dir, "data_labeler_agent")
if data_agent_dir not in sys.path:
    sys.path.insert(0, data_agent_dir)

try:
    from data_labeler_agent.break_labeler_agent.break_labeler_agent import BreakLabelerAgent
except ImportError:
    from break_labeler_agent.break_labeler_agent import BreakLabelerAgent

try:
    from data_labeler_agent.solution_labeler_agent.solution_labeler_agent import process_breaks_to_solutions
except ImportError:
    from solution_labeler_agent.solution_labeler_agent import process_breaks_to_solutions

try:
    from data_labeler_agent.rule_labeler.scripts.make_error_prediction import (
        build_x_symptoms,
        map_label,
        make_error_prediction_row,
        _prepare_rules_with_normalizer,
    )
except ImportError:
    from rule_labeler.scripts.make_error_prediction import (
        build_x_symptoms,
        map_label,
        make_error_prediction_row,
        _prepare_rules_with_normalizer,
    )

try:
    from config import DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY
except Exception:
    DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY = 3

try:
    from data_labeler_agent.diagnostic_agent.diagnostic_agent import (
        predict_diagnostics,
        predict_diagnostics_batch,
    )
except ImportError:
    try:
        from data_labeler.diagnostic_agent.diagnostic_agent import (
            predict_diagnostics,
            predict_diagnostics_batch,
        )
    except ImportError:
        try:
            from diagnostic_agent.diagnostic_agent import (
                predict_diagnostics,
                predict_diagnostics_batch,
            )
        except ImportError:
            predict_diagnostics = None  # type: ignore
            predict_diagnostics_batch = None  # type: ignore


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _format_hms(seconds: float) -> str:
    total_seconds = int(max(0, round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_equipment(labels: Dict[str, Any]) -> Dict[str, str]:
    system_info = labels.get("system_info", {}) if isinstance(labels, dict) else {}
    return {
        "family": system_info.get("asset_family", "") or "",
        "subtype": system_info.get("asset_subtype", "") or "",
        "brand": system_info.get("brand", "") or "",
    }


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested dictionaries for CSV output and serialize complex lists.

    - Nested dicts are flattened using dot notation (e.g., equip.family)
    - Lists of primitives are joined by '; '
    - Lists with nested structures are JSON-serialized
    """
    flat: Dict[str, Any] = {}

    def _is_primitive(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) or value is None

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else str(k)
                _flatten(v, new_key)
        elif isinstance(obj, list):
            if all(_is_primitive(item) for item in obj):
                flat[prefix] = "; ".join("" if item is None else str(item) for item in obj)
            else:
                # complex list: keep as JSON
                flat[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            flat[prefix] = obj

    _flatten(record)
    return flat


def _clamp_conf(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _extract_prediction_conf(pred: Dict[str, Any]) -> float:
    """Extract confidence using base_score only; fallback to top-level confidence.

    Downstream capping is applied separately using the rule floor.
    """

    breakdown = pred.get("confidence_breakdown")
    if isinstance(breakdown, dict):
        base_score = breakdown.get("base_score")
        if isinstance(base_score, (int, float)):
            return max(0.0, min(1.0, float(base_score)))

    conf_value = pred.get("confidence")
    if isinstance(conf_value, (int, float)):
        return max(0.0, min(1.0, float(conf_value)))

    try:
        return max(0.0, min(1.0, float(conf_value)))  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _compute_rule_conf_floor(rules: Dict[str, Any], default_floor: float = 0.6) -> float:
    """Compute the minimum configured rule score to use as LLM confidence cap.

    If no explicit scores are present, fallback to ``default_floor``.
    """
    try:
        ordered = rules.get("rules") or rules.get("ordered_rules") or []
        floor = 1.0
        found_any = False
        for rule in ordered:
            if isinstance(rule, dict):
                if "score" in rule:
                    try:
                        s = float(rule.get("score", 1.0))
                    except Exception:
                        s = 1.0
                    floor = min(floor, s)
                    found_any = True
                else:
                    # Missing score defaults to 1.0; doesn't lower the floor
                    floor = min(floor, 1.0)
        if not found_any:
            return float(default_floor)
        floor = max(0.0, min(1.0, float(floor)))
        # Never let the LLM cap fall below the configured default floor—older rule
        # sets can include very low scores (e.g., 0.2) which would otherwise
        # suppress LLM results entirely.
        return max(float(default_floor), floor)
    except Exception:
        return float(default_floor)


def _setup_logger(run_output_dir: str) -> "logging.Logger":
    """Create a single root logger that writes to break_labeler.log and stdout.

    All module loggers (orchestrator, break labeler, solution labeler, etc.)
    will propagate into this root logger so we have one unified log file per run.
    """
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers across multiple invocations
    while logger.handlers:
        logger.handlers.pop()

    os.makedirs(run_output_dir, exist_ok=True)
    log_path = os.path.join(run_output_dir, "break_labeler.log")

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def _augment_with_diagnostics(
    records: Dict[str, Dict[str, Any]],
    rules: Dict[str, Any],
    norm_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply rule-based diagnostics and LLM fallback to enriched records."""

    rule_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"llm_attempted": 0, "llm_succeeded": 0, "llm_errors": []}

    llm_available = bool(predict_diagnostics_batch or predict_diagnostics)
    llm_requests: List[Tuple[str, Dict[str, Any]]] = []
    llm_results: Dict[str, Dict[str, Any]] = {}
    llm_errors: Dict[str, str] = {}
    per_post: List[Dict[str, Any]] = []

    rule_floor_conf = _compute_rule_conf_floor(rules)

    for post_id, rec in records.items():
        labels = rec.get("labels", {}) if isinstance(rec, dict) else {}
        post = rec.get("post", {}) if isinstance(rec, dict) else {}

        error_report = labels.get("error_report", {}) if isinstance(labels, dict) else {}
        if isinstance(labels, dict) and "error_report" not in labels:
            labels["error_report"] = error_report
        symptoms = error_report.get("symptoms", []) or []
        title = post.get("title", "") or ""
        body = post.get("body", "") or ""
        reddit_url = post.get("url", "") or ""
        subreddit = post.get("subreddit", "") or ""

        equip = _extract_equipment(labels)

        try:
            x_symptoms, provenance = build_x_symptoms(symptoms, title, body, norm_cfg)
        except Exception:
            x_symptoms, provenance = "", "unavailable"

        label_id, rule_conf, fired_rules = map_label(x_symptoms, equip, rules)
        rule_provenance = f"rules_v1::{provenance}"

        rule_row = make_error_prediction_row(
            post_id=post_id,
            x_symptoms=x_symptoms,
            equip=equip,
            label_id=label_id,
            sample_weight=1.0,
            ontology="diagnostics_v1",
            provenance=rule_provenance,
            fired_rules=fired_rules,
        )
        rule_row["rule_confidence"] = _clamp_conf(rule_conf)
        rule_rows.append(rule_row)

        is_unclear = label_id == "dx.other_or_unclear"
        need_llm = llm_available and (not fired_rules or is_unclear)
        if need_llm:
            stats["llm_attempted"] += 1
            llm_requests.append(
                (
                    post_id,
                    {
                        "post_id": post_id,
                        "title": title,
                        "body": body,
                        "symptoms": symptoms,
                        "equip": equip,
                    },
                )
            )

        rec["x_symptoms"] = x_symptoms
        per_post.append(
            {
                "post_id": post_id,
                "record": rec,
                "error_report": error_report,
                "rule_label": label_id,
                "rule_conf": rule_conf,
                "rule_provenance": rule_provenance,
                "fired_rules": fired_rules,
                "x_symptoms": x_symptoms,
                "title": title,
                "body": body,
                "equip": equip,
                "reddit_url": reddit_url,
                "subreddit": subreddit,
                "need_llm": need_llm,
            }
        )

    if llm_requests and llm_available:
        payloads = [payload for _, payload in llm_requests]
        batch_outputs: Optional[List[Dict[str, Any]]] = None

        if predict_diagnostics_batch is not None:
            try:
                batch_outputs = predict_diagnostics_batch(
                    payloads,
                    max_concurrency=DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY,
                    confidence_max=rule_floor_conf,
                )
            except Exception as batch_exc:  # pragma: no cover - defensive fallback
                batch_outputs = None
                stats.setdefault("llm_errors", []).append(
                    {"post_id": "batch", "error": str(batch_exc)}
                )

        if batch_outputs is not None:
            for idx, (post_id, _) in enumerate(llm_requests):
                if idx < len(batch_outputs):
                    llm_results[post_id] = batch_outputs[idx]
                else:
                    llm_errors[post_id] = "missing output from batch execution"
        else:
            for post_id, payload in llm_requests:
                try:
                    if predict_diagnostics is None:
                        raise RuntimeError("predict_diagnostics unavailable")
                    llm_results[post_id] = predict_diagnostics(
                        payload,
                        max_concurrency=DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY,
                        confidence_max=rule_floor_conf,
                    )  # type: ignore[arg-type]
                except Exception as exc:  # pragma: no cover - defensive for missing creds
                    llm_errors[post_id] = str(exc)

    llm_runtime_available = llm_available

    for entry in per_post:
        post_id = entry["post_id"]
        rec = entry["record"]
        error_report = entry["error_report"]

        final_label = entry["rule_label"]
        final_conf = _clamp_conf(entry["rule_conf"])
        final_provenance = entry["rule_provenance"]
        llm_payload: Optional[Dict[str, Any]] = None
        conflict_entry: Optional[Dict[str, Any]] = None

        if entry["need_llm"]:
            if post_id in llm_results:
                llm_payload = llm_results[post_id]
            elif post_id in llm_errors:
                err_msg = llm_errors[post_id]
                stats.setdefault("llm_errors", []).append(
                    {"post_id": post_id, "error": err_msg}
                )
                lower_msg = err_msg.lower()
                permanent_cred_error = (
                    "gemini_api_key" in lower_msg
                    or "invalid api key" in lower_msg
                    or "unauthorized" in lower_msg
                )
                if permanent_cred_error:
                    llm_runtime_available = False
                llm_payload = {"error": err_msg}
            elif llm_runtime_available:
                missing_msg = "missing LLM output"
                stats.setdefault("llm_errors", []).append(
                    {"post_id": post_id, "error": missing_msg}
                )
                llm_payload = {"error": missing_msg}
            else:
                llm_payload = {"error": "diagnostic agent unavailable"}

        if isinstance(llm_payload, dict) and "predictions" in llm_payload:
            preds = llm_payload.get("predictions", [])
        elif llm_payload is not None:
            preds = []
        else:
            preds = []

        if llm_payload is not None and not preds:
            llm_payload.setdefault("predictions", [])  # type: ignore[union-attr]

        if preds:
            top = preds[0]
            maybe_label = (top.get("label_id") or "").strip()
            if maybe_label:
                # Clamp LLM confidence to the rule floor (e.g., 0.6) upper bound
                llm_conf_raw = _extract_prediction_conf(top)
                llm_conf = min(rule_floor_conf, _clamp_conf(llm_conf_raw))
                conflict_entry = (
                    {
                        "source": "rules_vs_llm",
                        "rule": {
                            "label": entry["rule_label"],
                            "confidence": _clamp_conf(entry["rule_conf"]),
                        },
                        "llm": {
                            "label": maybe_label,
                            "confidence": llm_conf,
                            "provenance": llm_payload.get("provenance", "llm_v1"),
                        },
                    }
                    if maybe_label != entry["rule_label"]
                    else None
                )

                if llm_conf > final_conf:
                    final_label = maybe_label
                    final_conf = llm_conf
                    final_provenance = llm_payload.get("provenance", "llm_v1")
                    stats["llm_succeeded"] += 1
                    if conflict_entry is not None:
                        conflict_entry["chosen"] = "llm"
                elif conflict_entry is not None:
                    conflict_entry["chosen"] = "rules"

        if conflict_entry is not None:
            error_report.setdefault("conflicts", []).append(conflict_entry)

        rec["diagnostics"] = {
            "rule_based": {
                "label_id": entry["rule_label"],
                "confidence": _clamp_conf(entry["rule_conf"]),
                "fired_rules": entry["fired_rules"],
                "provenance": entry["rule_provenance"],
            },
            "llm": llm_payload,
            "final": {
                "label_id": final_label,
                "confidence": final_conf,
                "provenance": final_provenance,
                "source": "llm" if final_provenance.startswith("llm") else "rules",
            },
        }

        final_rows.append(
            {
                "post_id": post_id,
                "x_symptoms": entry["x_symptoms"],
                "x_post": f"{entry['title'].strip()}\n\n{entry['body'].strip()}".strip(),
                "equip": entry["equip"],
                # Weight equals selected final confidence (rule or LLM)
                "y_diag": [[final_label, float(_clamp_conf(final_conf))]],
                "provenance": final_provenance,
                "rule_label": entry["rule_label"],
                "rule_confidence": _clamp_conf(entry["rule_conf"]),
                "rule_fired_rules": entry["fired_rules"],
                "reddit_url": entry["reddit_url"],
                "subreddit": entry["subreddit"],
            }
        )

    return records, rule_rows, final_rows, stats


def process_reddit_data_to_solutions(
    reddit_data_file: str,
    output_dir: str = "output",
    subreddits: Optional[list] = None,
    solution_max_concurrency: Optional[int] = None,
) -> dict:
    """
    Full pipeline: Reddit data → Break labels → Solutions
    
    Args:
        reddit_data_file: Path to reddit_research_data_*.json
        output_dir: Output directory for intermediate and final files
        subreddits: Optional list of subreddit names to filter
        solution_max_concurrency: Optional override for concurrent LLM calls
        
    Returns:
        Dict with paths to intermediate and final output files
    """
    t0 = time.time()
    total_posts_processed = 0

    base_output_dir = os.path.abspath(output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    run_dt = datetime.now()
    run_date = run_dt.strftime("%Y-%m-%d")
    timestamp = run_dt.strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(base_output_dir, run_date)
    os.makedirs(run_output_dir, exist_ok=True)

    logger = _setup_logger(run_output_dir)
    logger.info("🚀 Starting data labeling pipeline...")
    logger.info("📥 Input: %s", reddit_data_file)
    logger.info("📁 Run output directory: %s", run_output_dir)
    
    # Step 1: Break labeling
    logger.info("\n📊 Step 1: Break labeling...")
    step1_start = time.time()
    break_agent = BreakLabelerAgent(output_dir=run_output_dir)
    break_result = break_agent.label_from_json_file(
        reddit_data_file,
        subreddits=subreddits
    )
    
    if not break_result.get("success"):
        logger.error("Break labeling failed: %s", break_result.get('error'))
        raise Exception(f"Break labeling failed: {break_result.get('error')}")
    
    break_labels_file = break_result["output_file"]
    logger.info("✅ Break labels: %s", break_labels_file)
    logger.info("⏱️ Step 1 duration: %s", _format_hms(time.time() - step1_start))
    
    # Step 2: Solution extraction
    logger.info("\n💡 Step 2: Solution extraction...")
    solutions_file = os.path.join(run_output_dir, f"solutions_{timestamp}.json")
    step2_start = time.time()
    solutions_path = process_breaks_to_solutions(
        raw_file=reddit_data_file,
        labels_file=break_labels_file,
        out_file=solutions_file,
        max_concurrency=solution_max_concurrency,
        output_dir=base_output_dir,
    )
    logger.info("✅ Solutions: %s", solutions_path)
    logger.info("⏱️ Step 2 duration: %s", _format_hms(time.time() - step2_start))

    solutions_doc = _load_json(solutions_path) if os.path.exists(solutions_path) else {}
    try:
        total_posts_processed = len(solutions_doc) if isinstance(solutions_doc, (dict, list)) else 0
    except Exception:
        total_posts_processed = 0
    logger.info("🧮 Posts processed: %s", total_posts_processed)

    # Step 3: Rule-based diagnostics
    logger.info("\n🧠 Step 3: Rule-based diagnostics...")
    step3_start = time.time()
    rules_path = os.path.join(current_dir, "rule_labeler", "meta", "rules_v1.json")
    norm_cfg_path = os.path.join(current_dir, "rule_labeler", "scripts", "make_error_prediction_config.json")

    rules_cfg = _load_json(rules_path) if os.path.exists(rules_path) else {}
    norm_cfg = _load_json(norm_cfg_path) if os.path.exists(norm_cfg_path) else {}
    if not norm_cfg:
        norm_cfg = {"aliases": [], "unit_patterns": [], "phrase_compact": [], "max_len": 1500}
    prepared_rules = _prepare_rules_with_normalizer(rules_cfg, norm_cfg) if rules_cfg else {}

    augmented, rule_rows, final_rows, stats = _augment_with_diagnostics(solutions_doc, prepared_rules, norm_cfg)

    rule_rows_file = os.path.join(run_output_dir, f"rule_predictions_{timestamp}.json")
    diagnostics_file = os.path.join(run_output_dir, f"solutions_with_diagnostics_{timestamp}.json")
    final_dataset_file = os.path.join(run_output_dir, f"diagnostic_dataset_{timestamp}.json")

    _write_json(rule_rows_file, rule_rows)
    _write_json(diagnostics_file, augmented)
    _write_json(final_dataset_file, final_rows)

    logger.info("✅ Rule predictions: %s", rule_rows_file)
    logger.info("⏱️ Step 3 duration: %s", _format_hms(time.time() - step3_start))

    # Step 3b: Export diagnostic dataset to CSV
    try:
        # Expand y_diag ([[label, score]]) into flat CSV columns: diagnosis | weight
        transformed_rows: List[Dict[str, Any]] = []
        for row in final_rows:
            row_out = dict(row)
            y_diag = row_out.pop("y_diag", None)
            diagnosis = ""
            weight_value: Optional[float] = None
            if isinstance(y_diag, list) and y_diag:
                first = y_diag[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    diagnosis = str(first[0]) if first[0] is not None else ""
                    try:
                        weight_value = float(first[1]) if first[1] is not None else None
                    except Exception:
                        weight_value = None
            row_out["diagnosis"] = diagnosis
            row_out["weight"] = weight_value if weight_value is not None else ""
            transformed_rows.append(row_out)

        flattened_rows = [_flatten_record_for_csv(r) for r in transformed_rows]
        fieldnames = sorted({key for row in flattened_rows for key in row.keys()})
        final_dataset_csv = os.path.join(run_output_dir, f"diagnostic_dataset_{timestamp}.csv")
        with open(final_dataset_csv, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_rows)
        logger.info("✅ Diagnostic dataset CSV: %s", final_dataset_csv)
    except Exception as csv_exc:
        logger.warning("⚠️ Failed to write diagnostic dataset CSV: %s", csv_exc)

    # Step 4: Final diagnostic agent summary
    if stats.get("llm_attempted"):
        logger.info(
            "🤖 Final diagnostic agent attempted %s posts (success: %s)",
            stats["llm_attempted"],
            stats.get("llm_succeeded", 0),
        )
    if stats.get("llm_errors"):
        logger.warning("⚠️ Final diagnostic agent errors: %s", len(stats["llm_errors"]))

    logger.info("✅ Solutions + diagnostics: %s", diagnostics_file)
    logger.info("✅ Final dataset: %s", final_dataset_file)
    logger.info("⏳ Total runtime: %s", _format_hms(time.time() - t0))
    logger.info("🧮 Total posts processed: %s", total_posts_processed)
    logger.info("\n🎉 Pipeline complete!")

    return {
        "reddit_data": reddit_data_file,
        "break_labels": break_labels_file,
        "solutions": solutions_path,
        "rule_predictions": rule_rows_file,
        "solutions_with_diagnostics": diagnostics_file,
        "final_dataset": final_dataset_file,
        "final_dataset_csv": final_dataset_csv if 'final_dataset_csv' in locals() else None,
        "output_directory": run_output_dir,
        "orchestrator_log": os.path.join(run_output_dir, "data_labeler_orchestrator.log"),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Labeler Agent - Full pipeline")
    parser.add_argument("input", help="Path to reddit_research_data_*.json")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--subs", help="Comma-separated subreddit names to include")
    parser.add_argument(
        "--solution-max-concurrency",
        type=int,
        default=None,
        help="Optional override for concurrent LLM calls",
    )
    
    args = parser.parse_args()
    
    subs = [s.strip() for s in args.subs.split(",")] if args.subs else None
    
    try:
        result = process_reddit_data_to_solutions(
            reddit_data_file=args.input,
            output_dir=args.output_dir,
            subreddits=subs,
            solution_max_concurrency=args.solution_max_concurrency,
        )
        print(f"\n📋 Output files:")
        for key, path in result.items():
            print(f"  {key}: {path}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

