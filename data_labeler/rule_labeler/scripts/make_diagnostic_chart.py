## Script to pull latest diagnostic chart

import sys
from pathlib import Path
from typing import Dict, List, Set
import json

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import GOLDEN_DIAGNOSTIC_CHART_PATH  # noqa: E402
from data_labeler.rule_labeler.scripts.make_error_prediction import normalize  # noqa: E402

# Paths to meta files
DIAGNOSTICS_META_PATH = project_root / "data_labeler/rule_labeler/meta/diagnostics_v1.json"
RULES_META_PATH = project_root / "data_labeler/rule_labeler/meta/rules_v1.json"
NORMALIZER_CONFIG_PATH = project_root / "data_labeler/rule_labeler/scripts/make_error_prediction_config.json"

def get_golden_diagnostic_chart(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def update_diagnostics_meta(diagnostic_chart: dict) -> None:
    """Update diagnostics_v1.json with the latest diagnostic IDs and metadata from golden set."""
    # Read current diagnostics_v1.json (create base structure if missing)
    if DIAGNOSTICS_META_PATH.exists():
        with open(DIAGNOSTICS_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"version": 1}

    diagnostics = diagnostic_chart.get("diagnostics", []) or []

    labels: List[str] = []
    label_details: Dict[str, Dict[str, str]] = {}
    for diag in diagnostics:
        diag_id = diag.get("diagnostic_id")
        if not diag_id:
            continue
        labels.append(diag_id)
        label_details[diag_id] = {
            "description": diag.get("description")
            or diag.get("specific_diagnostic_name")
            or "",
            "typical_effect_on_operation": diag.get("typical_effect_on_operation", ""),
        }

    data["labels"] = labels
    data["label_details"] = label_details

    with open(DIAGNOSTICS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Updated {DIAGNOSTICS_META_PATH} with {len(labels)} diagnostic labels "
        "and descriptions"
    )


def _load_normalizer_cfg() -> dict:
    with open(NORMALIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _add_phrase(
    diag_id: str,
    phrase: str,
    norm_cfg: dict,
    phrases_by_diag: Dict[str, List[str]],
    seen_by_diag: Dict[str, Set[str]],
) -> None:
    if not phrase:
        return
    normalized, _ = normalize(str(phrase), norm_cfg)
    normalized = normalized.strip()
    if not normalized:
        return
    seen = seen_by_diag.setdefault(diag_id, set())
    if normalized in seen:
        return
    seen.add(normalized)
    phrases_by_diag.setdefault(diag_id, []).append(normalized)


def update_rules_meta(diagnostic_chart: dict) -> None:
    """Regenerate rules_v1.json from the golden diagnostic chart with normalized phrases."""
    diagnostics_by_id = {
        d["diagnostic_id"]: d for d in diagnostic_chart.get("diagnostics", [])
    }
    norm_cfg = _load_normalizer_cfg()

    phrases_by_diag: Dict[str, List[str]] = {}
    seen_by_diag: Dict[str, Set[str]] = {}
    scores_by_diag: Dict[str, List[float]] = {}
    mapping_ids_by_diag: Dict[str, List[str]] = {}

    def ensure_diag(diag_id: str) -> None:
        phrases_by_diag.setdefault(diag_id, [])
        seen_by_diag.setdefault(diag_id, set())
        scores_by_diag.setdefault(diag_id, [])
        mapping_ids_by_diag.setdefault(diag_id, [])

    # Start with loose + canonical symptom mappings
    for mapping in diagnostic_chart.get("symptom_mappings", []):
        diag_id = mapping["diagnostic_id"]
        ensure_diag(diag_id)
        mapping_ids_by_diag[diag_id].append(mapping.get("mapping_id", ""))
        scores_by_diag[diag_id].append(float(mapping.get("likelihood_relative", 0.0)))
        _add_phrase(diag_id, mapping.get("loose_symptom_example"), norm_cfg, phrases_by_diag, seen_by_diag)
        _add_phrase(diag_id, mapping.get("canonical_symptom"), norm_cfg, phrases_by_diag, seen_by_diag)

    # Enrich with additional diagnostic fields
    extra_fields = (
        "canonical_symptoms",
        "common_root_causes",
        "technician_observation_phrases",
        "technician_observation_codes",
    )
    for diag_id, diag in diagnostics_by_id.items():
        ensure_diag(diag_id)
        for field in extra_fields:
            for phrase in diag.get(field, []) or []:
                _add_phrase(diag_id, phrase, norm_cfg, phrases_by_diag, seen_by_diag)

    rules: List[dict] = []
    for diag_id in sorted(phrases_by_diag.keys()):
        phrases = phrases_by_diag.get(diag_id, [])
        if not phrases:
            continue
        score_candidates = scores_by_diag.get(diag_id, [])
        score = max(score_candidates) if score_candidates else 0.3
        diag = diagnostics_by_id.get(diag_id, {})
        notes_parts: List[str] = []
        mapping_ids = [mid for mid in mapping_ids_by_diag.get(diag_id, []) if mid]
        if mapping_ids:
            notes_parts.append(f"auto from mapping_ids={','.join(mapping_ids)}")
        if diag.get("specific_diagnostic_name"):
            notes_parts.append(f"specific={diag['specific_diagnostic_name']}")
        notes = "; ".join(notes_parts) if notes_parts else "auto from diagnostics_v3"
        rules.append(
            {
                "label": diag_id,
                "any": phrases,
                "all": [],
                "score": float(score),
                "notes": notes,
            }
        )

    data = {
        "rules": rules,
        "fallback": "dx.other_or_unclear",
    }
    with open(RULES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated {RULES_META_PATH} with {len(rules)} rules")


def update_system_types(diagnostic_chart: dict) -> None:
    """Generate system_types.json from golden_set_v3 diagnostics.

    Structure:
    {
      "system_types": [...unique system_types...],
      "by_diagnostic_id": {"dx.*": [system_types...]}
    }
    """
    diagnostics = diagnostic_chart.get("diagnostics", [])

    unique_system_types: set[str] = set()
    by_diag: dict[str, list[str]] = {}

    for diag in diagnostics:
        diag_id = diag.get("diagnostic_id")
        system_types = diag.get("system_types") or []

        for st in system_types:
            unique_system_types.add(st)

        if diag_id:
            by_diag[diag_id] = list(system_types)

    data = {
        "system_types": sorted(unique_system_types),
        "by_diagnostic_id": by_diag,
    }

    # Write system_types.json into the shared meta folder so all agents use
    # the same canonical location:
    #   data_labeler/rule_labeler/meta/system_types.json
    meta_dir = project_root / "data_labeler" / "rule_labeler" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    system_types_path = meta_dir / "system_types.json"
    with open(system_types_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated {system_types_path} with {len(data['system_types'])} system types")

def main():
    # Convert relative path to absolute from project root
    path = project_root / GOLDEN_DIAGNOSTIC_CHART_PATH
    diagnostic_chart = get_golden_diagnostic_chart(str(path))
    update_diagnostics_meta(diagnostic_chart)
    update_rules_meta(diagnostic_chart)
    update_system_types(diagnostic_chart)


if __name__ == "__main__":
    main()