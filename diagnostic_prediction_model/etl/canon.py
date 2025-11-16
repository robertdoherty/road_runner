# diagnostic_prediction_model/canon.py

import re
import json
import hashlib
from pathlib import Path

def canonicalize_fields(symptoms: str) -> list[str]:
    """Canonicalize a symptoms string.

    Args:
        xsymptoms: The symptoms string to canonicalize.

    Returns:
        The canonicalized symptoms list.
    """
    parts = [p.strip().lower() for p in symptoms.split(";")]
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        p = p.replace("/", "_")
        if p:
            out.append(p.replace(" ", "_"))
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); dedup.append(t)
    return dedup


def canonicalize_equip(equip: dict) -> dict:
    """Canonicalize equipment fields similar to symptoms (lowercase, underscores).

    Args:
        equip: Dict possibly containing 'system_type', 'subtype', 'brand'.

    Returns:
        Dict with canonicalized string values for 'system_type', 'subtype', 'brand'.
    """
    def norm(value: str) -> str:
        if not isinstance(value, str):
            return ""
        # similar to canonicalize_fields per-token normalization
        token = value.strip().lower()
        token = re.sub(r"[.,!?]+$", "", token)           # drop trailing punctuation
        token = re.sub(r"\s+", " ", token).strip()       # collapse whitespace
        token = token.replace("-", "_")                   # hyphens -> underscores
        token = token.replace(" ", "_")                   # spaces -> underscores
        return token

    return {
        "system_type": norm(equip.get("system_type", "")),
        # For training, treat system_subtype as the canonical subtype source.
        "subtype": norm(equip.get("system_subtype", "")),
        "brand": norm(equip.get("brand", "")),
        "model_family_id": norm(equip.get("model_family_id", "")),
        "model_text": norm(equip.get("model_text", "")),
    }


def split_from_id(post_id: str) -> str:
    """Deterministically assign each post to train/val/test.
    Uses a hash of post_id so the split is stable between runs.
    """
    h = int(hashlib.md5(post_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"

def build_training_data(path_to_json: str) -> dict:
    """Build training data from a JSON file.

    Args:
        path_to_json: The path to the JSON file.

    Returns:
        The training data.
    """
    diagnostic_dataset = json.load(open(path_to_json, 'r'))
    
    # Set up data directory relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate empty sets and training jsonl
    symptom_set, system_type_set, subtype_set, brand_set, diag_set = set(), set(), set(), set(), set()
    out_files = {
        "train": open(data_dir / "train.jsonl","w"),
        "val":   open(data_dir / "val.jsonl","w"),
        "test":  open(data_dir / "test.jsonl","w"),
    }
    ## Go through all issues and add to sets
    filtered_count = 0
    for issue in diagnostic_dataset:
        # Canonicalize symptoms
        issue['symptoms_canon'] = canonicalize_fields(issue.get('x_symptoms', ''))
        symptom_set.update(issue['symptoms_canon'])

        # Canonicalize equipment and WRITE BACK so JSONL has canonical values
        equip_canon = canonicalize_equip(issue.get('equip', {}))
        issue['equip'] = equip_canon
        if equip_canon['system_type']:
            system_type_set.add(equip_canon['system_type'])
        if equip_canon['subtype']:
            subtype_set.add(equip_canon['subtype'])
        if equip_canon['brand']:
            brand_set.add(equip_canon['brand'])

        # Diagnostic label (as-is from dataset entry)
        diag_label = None
        try:
            diag_label = issue['y_diag'][0][0]
        except Exception:
            diag_label = None
        
        # Filter out unclear diagnostics
        if diag_label == "dx.other_or_unclear":
            filtered_count += 1
            continue
        
        if diag_label:
            diag_set.add(diag_label)

        # Split from id to train/val/test and write out
        issue["split"] = split_from_id(issue["post_id"])
        out_files[issue["split"]].write(json.dumps(issue) + "\n")
    
    for f in out_files.values(): f.close()

    def to_idx_map(vals): return {v:i for i,v in enumerate(sorted(vals))}

    # Ensure unknown tokens exist for sparse fields
    subtype_set.add("<unk_subtype>")
    brand_set.add("<unk_brand>")
    system_type_set.add("<unk_system_type>")

    vocabs = {
        "symptom2id": to_idx_map(symptom_set),
        "system_type2id":  to_idx_map(system_type_set),
        "subtype2id": to_idx_map(subtype_set),
        "brand2id":   to_idx_map(brand_set),
        "diag2id":    to_idx_map(diag_set),
        "version": 1
    }
    
    json.dump(vocabs, open(data_dir / "vocabs.json","w"), indent=2)
    
    # Return statistics for reporting
    train_count = sum(1 for issue in diagnostic_dataset if issue.get('split') == 'train')
    val_count = sum(1 for issue in diagnostic_dataset if issue.get('split') == 'val')
    test_count = sum(1 for issue in diagnostic_dataset if issue.get('split') == 'test')
    
    return {
        "total": len(diagnostic_dataset),
        "filtered": filtered_count,
        "used": train_count + val_count + test_count,
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "vocab_sizes": {
            "symptoms": len(symptom_set),
            "system_types": len(system_type_set),
            "subtypes": len(subtype_set),
            "brands": len(brand_set),
            "diagnostics": len(diag_set)
        },
        "output_dir": str(data_dir)
    }


if __name__ == "__main__":
    # Use the most recent diagnostic dataset
    # Navigate to project root (3 levels up from this file)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    input_file = project_root / "output" / "2025-11-03" / "diagnostic_dataset_2025-11-03_20-59-20.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        print(f"Project root: {project_root}")
        exit(1)
    
    print(f"Building training data from: {input_file}\n")
    stats = build_training_data(str(input_file))
    
    print(f"✓ Created training data from {stats['total']} total samples")
    print(f"  - Filtered out: {stats['filtered']} (dx.other_or_unclear)")
    print(f"  - Used: {stats['used']} samples")
    print(f"    • Train: {stats['train']} samples")
    print(f"    • Val: {stats['val']} samples")
    print(f"    • Test: {stats['test']} samples")
    print(f"\n✓ Vocabulary sizes:")
    print(f"  - Symptoms: {stats['vocab_sizes']['symptoms']}")
    print(f"  - System Types: {stats['vocab_sizes']['system_types']}")
    print(f"  - Subtypes: {stats['vocab_sizes']['subtypes']}")
    print(f"  - Brands: {stats['vocab_sizes']['brands']}")
    print(f"  - Diagnostics: {stats['vocab_sizes']['diagnostics']}")
    print(f"\n✓ Output files written to: {stats['output_dir']}/")
