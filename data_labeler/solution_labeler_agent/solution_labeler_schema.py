from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, confloat, validator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_system_types() -> List[str]:
    path = _repo_root() / "data_labeler" / "rule_labeler" / "meta" / "system_types.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            values = data.get("system_types") or []
            if isinstance(values, list):
                return [v for v in values if isinstance(v, str)]
    except Exception as e:
        raise RuntimeError(f"Failed to load system types from {path}: {e}")
    return []


SYSTEM_TYPE_CHOICES: List[str] = sorted(set(_load_system_types()))
DEFAULT_SYSTEM_TYPE = "other_or_unclear"
if DEFAULT_SYSTEM_TYPE not in SYSTEM_TYPE_CHOICES:
    SYSTEM_TYPE_CHOICES.append(DEFAULT_SYSTEM_TYPE)

ProvenanceLiteral = Literal["op_comment", "op_edit", "commenter"]


class PartSpec(BaseModel):
    model: str = ""
    part: str = ""
    sku: str = ""


class SolutionReport(BaseModel):
    summary: str = ""
    steps: List[str] = Field(default_factory=list)
    parts_needed: List[PartSpec] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0) = 0.0


class ErrorReportEvidenceRefs(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    error_codes: List[str] = Field(default_factory=list)


class ErrorReportProvenance(BaseModel):
    symptoms: Optional[ProvenanceLiteral] = None
    error_codes: Optional[ProvenanceLiteral] = None


class ErrorReportFieldConfidence(BaseModel):
    symptoms: confloat(ge=0.0, le=1.0) = 0.0
    error_codes: confloat(ge=0.0, le=1.0) = 0.0


class ErrorReportDelta(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    error_codes: List[str] = Field(default_factory=list)
    evidence_refs_by_field: ErrorReportEvidenceRefs = Field(default_factory=ErrorReportEvidenceRefs)
    provenance_by_field: ErrorReportProvenance = Field(default_factory=ErrorReportProvenance)
    field_confidence_by_field: ErrorReportFieldConfidence = Field(default_factory=ErrorReportFieldConfidence)


class SystemInfoEvidenceRefs(BaseModel):
    system_type: List[str] = Field(default_factory=list)
    asset_subtype: List[str] = Field(default_factory=list)
    brand: List[str] = Field(default_factory=list)
    model_text: List[str] = Field(default_factory=list)
    model_family_id: List[str] = Field(default_factory=list)


class SystemInfoProvenance(BaseModel):
    system_type: Optional[ProvenanceLiteral] = None
    asset_subtype: Optional[ProvenanceLiteral] = None
    brand: Optional[ProvenanceLiteral] = None
    model_text: Optional[ProvenanceLiteral] = None
    model_family_id: Optional[ProvenanceLiteral] = None


class SystemInfoFieldConfidence(BaseModel):
    system_type: confloat(ge=0.0, le=1.0) = 0.0
    asset_subtype: confloat(ge=0.0, le=1.0) = 0.0
    brand: confloat(ge=0.0, le=1.0) = 0.0
    model_text: confloat(ge=0.0, le=1.0) = 0.0
    model_family_id: confloat(ge=0.0, le=1.0) = 0.0


class SystemInfoDelta(BaseModel):
    system_type: str = DEFAULT_SYSTEM_TYPE
    asset_subtype: str = ""
    brand: str = ""
    model_text: str = ""
    model_family_id: str = ""
    evidence_refs_by_field: SystemInfoEvidenceRefs = Field(default_factory=SystemInfoEvidenceRefs)
    provenance_by_field: SystemInfoProvenance = Field(default_factory=SystemInfoProvenance)
    field_confidence_by_field: SystemInfoFieldConfidence = Field(default_factory=SystemInfoFieldConfidence)

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


class EnrichmentBlock(BaseModel):
    error_report_delta: ErrorReportDelta = Field(default_factory=ErrorReportDelta)
    system_info_delta: SystemInfoDelta = Field(default_factory=SystemInfoDelta)


class SolutionLabelerOutput(BaseModel):
    post_id: str
    solution_report: SolutionReport
    enrichment: EnrichmentBlock = Field(default_factory=EnrichmentBlock)


parser = PydanticOutputParser(pydantic_object=SolutionLabelerOutput)

