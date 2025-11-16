"""Utilities for error prediction preprocessing.

Provides functions to normalize text, extract fields from labeled solution
data, and construct a canonical symptoms string for each post.
"""

from typing import Any, Dict, List
import json
import re
import os

PUNCT_RE = re.compile(r"[, :?!()<>]")
WS_RE    = re.compile(r"\s+")

def normalize(text: str, cfg: dict) -> tuple[str, list[str]]:
    """Normalize a text string using regex-based rules from config.

    The normalization pipeline lowercases, collapses whitespace, applies alias
    and unit/phrase compaction rules, strips punctuation while preserving
    placeholder tokens like ``<brand_...>``, and trims to a maximum length.

    Args:
        text: Raw input text to normalize.
        cfg: Config with keys ``aliases``, ``unit_patterns``, ``phrase_compact``,
            and optional ``max_len``.

    Returns:
        A tuple ``(normalized_text, fired_rule_tags)``.
    """
    fired = []
    s = text.lower()
    s = s.replace("\u200b", " ").replace("\n", " ")
    s = WS_RE.sub(" ", s).strip()

    # aliases
    for rule in cfg["aliases"]:
        before = s
        s = re.sub(rule["pattern"], rule["repl"], s)
        if s != before: fired.append(f"alias:{rule['repl']}")

    # units
    for rule in cfg["unit_patterns"]:
        before = s
        s = re.sub(rule["pattern"], rule["format"], s)
        if s != before: fired.append(f"unit:{rule['name']}")

    # phrases
    for rule in cfg["phrase_compact"]:
        before = s
        s = re.sub(rule["pattern"], rule["repl"], s)
        if s != before: fired.append(f"phrase:{rule['repl']}")

    # punctuation (preserve <tokens>)
    def _safe_punct(m):
        return " "
    s = PUNCT_RE.sub(_safe_punct, s)
    s = WS_RE.sub(" ", s).strip()

    # length cap
    if len(s) > cfg.get("max_len", 1500):
        s = s[:cfg["max_len"]].rsplit(" ", 1)[0]

    return s, fired


def extract_post_fields(solutions_json_path: str) -> List[Dict[str, Any]]:
    """Extract post id, symptoms, and equipment info from a solutions JSON.

    Args:
        solutions_json_path: Path to the labeled solutions JSON file.

    Returns:
        A list of dictionaries with keys:
        - ``post_id`` (str)
        - ``symptoms`` (List[str])
        - ``title`` (str)
        - ``body`` (str)
        - ``equip`` (dict with ``system_type``, ``system_subtype``, ``brand``)
    """

    with open(solutions_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    extracted: List[Dict[str, Any]] = []

    for obj in data.values():
        labels: Dict[str, Any] = obj.get("labels", {})
        error_report: Dict[str, Any] = labels.get("error_report", {})
        system_info: Dict[str, Any] = labels.get("system_info", {})
        post_blob: Dict[str, Any] = obj.get("post", {})

        post_id: str = obj.get("post_id", "")
        symptoms: List[str] = error_report.get("symptoms", []) or []
        title: str = post_blob.get("title", "") or ""
        body: str = post_blob.get("body", "") or ""

        equip: Dict[str, Any] = {
            "system_type": system_info.get("system_type", ""),
            "system_subtype": system_info.get("system_subtype", "") or system_info.get("asset_subtype", ""),
            "brand": system_info.get("brand", ""),
        }

        extracted.append({
            "post_id": post_id,
            "symptoms": symptoms,
            "title": title,
            "body": body,
            "equip": equip,
        })

    return extracted



def build_x_symptoms(symptoms: List[str], title: str, body: str, cfg: dict) -> tuple[str, str]:
    """Build a normalized symptoms string and record its provenance.

    Chooses the symptoms list if present; otherwise falls back to
    ``title + " " + body``. Performs normalization and post-normalization
    de-duplication for list inputs, then truncates to the configured max length.

    Args:
        symptoms: Candidate symptom phrases (may be empty).
        title: Post title used for fallback.
        body: Post body used for fallback.
        cfg: Normalization config (``aliases``, ``unit_patterns``, ``phrase_compact``,
            optional ``max_len``).

    Returns:
        A tuple ``(x_symptoms, provenance)`` where ``provenance`` is either
        ``"symptoms_list"`` or ``"title_body_fallback"``.
    """

    clean_symptoms = [s.strip() for s in (symptoms or []) if isinstance(s, str) and s.strip()]

    if clean_symptoms:
        provenance = "symptoms_list"
        assembled = "; ".join(clean_symptoms)
        used_list = True
    else:
        provenance = "title_body_fallback"
        t = title or ""
        b = body or ""
        assembled = f"{t} {b}".strip()
        used_list = False

    # Normalize
    normalized, _ = normalize(assembled, cfg)

    # Post-normalization cleanup for list-based inputs
    if used_list:
        parts = [p.strip() for p in normalized.split(";")]
        # remove empties and de-duplicate while preserving order
        seen = set()
        unique_parts = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                unique_parts.append(p)
        normalized = "; ".join(unique_parts)

    # Truncate to max length on a word boundary (extra safety beyond normalize)
    max_len = cfg.get("max_len", 1500)
    if len(normalized) > max_len:
        truncated = normalized[:max_len]
        normalized = truncated.rsplit(" ", 1)[0] if " " in truncated else truncated

    return normalized, provenance


def map_label(x_symptoms: str, equip: dict, rules: dict) -> tuple[str, float, list[str]]:
    """Map a normalized symptoms string to a label ID and confidence.

    Args:
        x_symptoms: Normalized symptoms string.
        equip: Equipment info dictionary.
        rules: Rules dictionary.

    Returns:
        A tuple ``(label_id, confidence, fired_rules)``.
    """
    def _to_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _equip_matches(rule_equip: dict, post_equip: dict) -> bool:
        if not rule_equip:
            return True
        post_equip = post_equip or {}
        for key in ("system_type", "subtype", "brand"):
            allowed = rule_equip.get(key)
            if allowed is None:
                continue
            post_key = "system_subtype" if key == "subtype" else key
            post_val = post_equip.get(post_key, "")
            if not post_val:
                return False
            if post_val not in _to_list(allowed):
                return False
        return True

    ordered = rules.get("rules") or rules.get("ordered_rules") or []
    hits: list[tuple[float, int, str]] = []  # (score, index, label)

    for idx, rule in enumerate(ordered):
        label_id = rule.get("id") or rule.get("label", f"rule_{idx}")
        phrases_all = _to_list(rule.get("all", rule.get("phrases_all", [])))
        phrases_any = _to_list(rule.get("any", []))

        if phrases_all and not all((p in x_symptoms) for p in phrases_all):
            continue
        if phrases_any and not any((p in x_symptoms) for p in phrases_any):
            continue
        if not _equip_matches(rule.get("equip", {}), equip):
            continue

        score = float(rule.get("score", 1.0))
        hits.append((score, idx, label_id))

    if hits:
        hits.sort(key=lambda t: (-t[0], t[1]))
        best_score, _best_idx, best_id = hits[0]
        # dedupe labels while preserving order
        seen_labels: set[str] = set()
        fired_rules: list[str] = []
        for _score, _idx, label in hits:
            if label not in seen_labels:
                seen_labels.add(label)
                fired_rules.append(label)
        return best_id, best_score, fired_rules

    fallback = rules.get("fallback", {})
    if isinstance(fallback, str):
        return fallback, 0.2, []
    return (
        fallback.get("id", "dx.other_or_unclear"),
        float(fallback.get("score", 0.2)),
        [],
    )

def make_error_prediction_row(
    post_id: str,
    x_symptoms: str,
    equip: dict,
    label_id: str,
    sample_weight: float,
    ontology: str,
    provenance: str,
    fired_rules: list[str] | None = None,
    fired_norm: list[str] | None = None,
) -> dict:
    """Assemble a single TaskA row.

    Args:
        post_id: Post identifier.
        x_symptoms: Normalized symptom text.
        equip: Equipment fields with ``system_type``, ``system_subtype``, ``brand``.
        label_id: Diagnostic label id to assign.
        sample_weight: Weight in ``[0.5, 1.0]``; will be clamped.
        ontology: Ontology/version string for labels.
        provenance: Source provenance for ``x_symptoms``.
        fired_rules: Optional list of rule ids that fired.
        fired_norm: Optional list of normalization events fired.

    Returns:
        A dictionary representing one training row with keys including
        ``post_id``, ``x_symptoms``, ``equip``, ``y_diag``, ``sample_weight``,
        ``ontology``, and ``provenance``; optionally ``fired_rules`` and
        ``fired_normalizer``.
    """
    row = {
        "post_id": post_id,
        "x_symptoms": x_symptoms,
        "equip": {
            "system_type": equip.get("system_type",""),
            "subtype": equip.get("subtype",""),
            "brand":  equip.get("brand",""),
        },
        "y_diag": [[label_id, 1.0]],
        "sample_weight": float(max(0.5, min(1.0, sample_weight))),
        "ontology": ontology,
        "provenance": provenance,
    }
    # Optional audit fields
    if fired_rules: row["fired_rules"] = fired_rules
    if fired_norm:  row["fired_normalizer"] = fired_norm
    return row


def append_jsonl(path: str, obj: dict) -> None:
    """Append one JSON object as a JSONL line.

    Args:
        path: Destination file path.
        obj: JSON-serializable object to append.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def update_golden_examples(
    store: dict[str, list[dict]],
    label_id: str,
    post_id: str,
    text: str,
    equip: dict,
    fired_rules: list[str],
    cap_per_label: int = 25,
) -> None:
    """Collect up to ``cap_per_label`` examples per label id.

    Args:
        store: Mapping of label id to list of example dicts.
        label_id: Label bucket to add the example under.
        post_id: Post identifier.
        text: Example text to store.
        equip: Equipment fields with ``system_type``, ``system_subtype``, ``brand``.
        fired_rules: Rule ids that matched for this example.
        cap_per_label: Maximum examples to retain per label.

    Returns:
        None.
    """
    bucket = store.setdefault(label_id, [])
    if len(bucket) < cap_per_label:
        # ensure hits unique
        unique_hits: list[str] = []
        seen: set[str] = set()
        for h in fired_rules:
            if h not in seen:
                seen.add(h)
                unique_hits.append(h)
        bucket.append({
            "post_id": post_id,
            "text": text,
            "equip": {k: equip.get(k,"") for k in ("system_type","subtype","brand")},
            "hits": unique_hits,
        })


def write_json(path: str, obj: dict) -> None:
    """Write a dictionary to a JSON file.

    Args:
        path: Destination file path.
        obj: JSON-serializable object to write.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_build(in_path: str, out_jsonl: str, rules: dict, norm_cfg: dict) -> None:
    """Build error prediction rows from an input solutions file.

    Reads posts, constructs normalized symptoms text, maps to a diagnostic label
    via rules, appends rows to a JSONL output, and collects a small set of
    golden examples per label for auditing.

    Args:
        in_path: Path to the labeled solutions JSON file.
        out_jsonl: Destination JSONL file to append rows into.
        rules: Rules configuration (ordered rules and fallback entry).
        norm_cfg: Normalization configuration for ``normalize``.

    Returns:
        None.
    """
    posts = extract_post_fields(in_path)
    gold = {}
    for p in posts:
        post_id = p["post_id"]; equip = p["equip"]
        x_symptoms, prov = build_x_symptoms(p["symptoms"], p.get("title",""), p.get("body",""), norm_cfg)
        label_id, conf, fired_rules = map_label(x_symptoms, equip, rules)
        row = make_error_prediction_row(
            post_id=post_id,
            x_symptoms=x_symptoms,
            equip=equip,
            label_id=label_id,
            sample_weight=1.0,  # or read from error_report if you pass it through
            ontology="diagnostics_v1",
            provenance=("rules_v1" if prov=="symptoms_list" else "rules_v1_fallback"),
            fired_rules=fired_rules
        )
        append_jsonl(out_jsonl, row)
        update_golden_examples(gold, label_id, post_id, x_symptoms, equip, fired_rules)

    write_json("error_prediction_model/gold/golden_examples.json", gold)


def _prepare_rules_with_normalizer(rules: dict, norm_cfg: dict) -> dict:
    """Return a copy of rules with phrases normalized via the same pipeline.

    Ensures that `all` and `any` rule phrases match the tokenization and
    normalization applied to x_symptoms, so substring checks are aligned.
    """
    prepared = {**rules}
    ordered = list(rules.get("rules") or rules.get("ordered_rules") or [])
    new_rules = []
    for rule in ordered:
        new_rule = dict(rule)
        for key in ("all", "any", "phrases_all"):
            if key in new_rule:
                phrases = new_rule.get(key) or []
                normed = []
                for p in phrases:
                    s, _ = normalize(str(p), norm_cfg)
                    if s:
                        normed.append(s)
                # de-duplicate preserving order
                seen = set()
                uniq = []
                for s in normed:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                if key == "phrases_all":
                    # map legacy key to `all`
                    new_rule["all"] = uniq
                    if key in new_rule:
                        new_rule.pop(key, None)
                else:
                    new_rule[key] = uniq
        new_rules.append(new_rule)
    if "rules" in prepared:
        prepared["rules"] = new_rules
    else:
        prepared["ordered_rules"] = new_rules
    # normalize fallback form to string or id/score (leave as-is)
    return prepared


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build error prediction dataset.")
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input solutions JSON (required)",
    )
    parser.add_argument(
        "--out",
        dest="out_jsonl",
        default="error_prediction_model/data/error_prediction.jsonl",
        help="Output JSONL path (default: error_prediction_model/data/error_prediction.jsonl)",
    )
    parser.add_argument(
        "--rules",
        dest="rules_path",
        default="error_prediction_model/meta/rules_v1.json",
        help="Rules JSON path (default: error_prediction_model/meta/rules_v1.json)",
    )
    parser.add_argument(
        "--norm",
        dest="norm_cfg_path",
        default="error_prediction_model/scripts/make_error_prediction_config.json",
        help="Normalizer config JSON path (default: error_prediction_model/scripts/make_error_prediction_config.json)",
    )

    args = parser.parse_args()

    with open(args.rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    with open(args.norm_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    prepared_rules = _prepare_rules_with_normalizer(rules, cfg)
    run_build(args.in_path, args.out_jsonl, prepared_rules, cfg)
