## Script to pull latest diagnostic chart

import sys
import os
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import GOLDEN_DIAGNOSTIC_CHART_PATH
import json

# Paths to meta files
DIAGNOSTICS_META_PATH = project_root / "data_labeler/rule_labeler/meta/diagnostics_v1.json"
RULES_META_PATH = project_root / "data_labeler/rule_labeler/meta/rules_v1.json"

def get_golden_diagnostic_chart(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def update_diagnostics_meta(diagnostic_chart: dict) -> None:
    """Update diagnostics_v1.json with the latest diagnostic IDs from golden set."""
    # Read current diagnostics_v1.json
    with open(DIAGNOSTICS_META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Update the labels field
    data["labels"] = [item["diagnostic_id"] for item in diagnostic_chart.get("diagnostics", [])]
    
    # Write back to file
    with open(DIAGNOSTICS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated {DIAGNOSTICS_META_PATH} with {len(data['labels'])} diagnostic labels")


def update_rules_meta(diagnostic_chart: dict) -> None:
    """Regenerate rules_v1.json from the golden diagnostic chart.

    For each symptom_mappings entry, create a simple rule:
    - label: diagnostic_id
    - any: [loose_symptom_example, canonical_symptom]
    - all: []
    - score: likelihood_relative
    """
    diagnostics_by_id = {
        d["diagnostic_id"]: d for d in diagnostic_chart.get("diagnostics", [])
    }

    rules: list[dict] = []
    for mapping in diagnostic_chart.get("symptom_mappings", []):
        diag_id = mapping["diagnostic_id"]
        diag = diagnostics_by_id.get(diag_id, {})

        any_phrases = [
            mapping["loose_symptom_example"],
            mapping["canonical_symptom"],
        ]

        rule: dict = {
            "label": diag_id,
            "any": any_phrases,
            "all": [],
            "score": float(mapping.get("likelihood_relative", 1.0)),
            "notes": f"auto from mapping_id={mapping['mapping_id']}",
        }

        rules.append(rule)

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