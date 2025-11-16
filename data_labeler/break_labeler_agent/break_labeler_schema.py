from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, confloat, validator
from langchain.output_parsers import PydanticOutputParser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_system_types() -> list[str]:
    path = _repo_root() / "data_labeler" / "rule_labeler" / "meta" / "system_types.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            values = data.get("system_types") or []
            if isinstance(values, list):
                return [v for v in values if isinstance(v, str)]
    except Exception as e:
        raise RuntimeError(f"Failed to load system types from {path}: {e}")


SYSTEM_TYPE_CHOICES: list[str] = sorted(set(_load_system_types()))
DEFAULT_SYSTEM_TYPE = "other_or_unclear"
if DEFAULT_SYSTEM_TYPE not in SYSTEM_TYPE_CHOICES:
    SYSTEM_TYPE_CHOICES.append(DEFAULT_SYSTEM_TYPE)

class ErrorReport(BaseModel):
    """Problem description and classification"""
    break_label: Literal["BREAK","NON_BREAK"]
    break_confidence: confloat(ge=0.0, le=1.0)
    symptoms: List[str] = Field(default_factory=list)
    symptoms_confidence: confloat(ge=0.0, le=1.0) = 0.0
    error_codes: List[str] = Field(default_factory=list)
    error_codes_confidence: confloat(ge=0.0, le=1.0) = 0.0

class SystemInfo(BaseModel):
    """Model/system identification and tagging"""
    system_type: str = DEFAULT_SYSTEM_TYPE
    system_type_confidence: confloat(ge=0.0, le=1.0) = 0.0
    asset_subtype: str = ""
    asset_subtype_confidence: confloat(ge=0.0, le=1.0) = 0.0
    brand: str = ""
    brand_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_text: str = ""
    model_text_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_family_id: str = ""
    model_family_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    indoor_model_id: str = ""
    indoor_model_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    outdoor_model_id: str = ""
    outdoor_model_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_resolution_confidence: confloat(ge=0.0, le=1.0) = 0.0
    has_images: bool = False

    @validator("system_type", pre=True, always=True)
    def _validate_system_type(cls, value: str) -> str:  # type: ignore[override]
        if not isinstance(value, str):
            return DEFAULT_SYSTEM_TYPE
        cleaned = value.strip()
        if not cleaned:
            return DEFAULT_SYSTEM_TYPE
        if cleaned not in SYSTEM_TYPE_CHOICES:
            return DEFAULT_SYSTEM_TYPE
        return cleaned

class BreakItem(BaseModel):
    id: str
    error_report: ErrorReport
    system_info: SystemInfo

class BreakOutput(BaseModel):
    results: List[BreakItem] = Field(..., min_length=1)

parser = PydanticOutputParser(pydantic_object=BreakOutput)
# You can add parser.get_format_instructions() to the prompt if you prefer structured guidance.
