from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatientProfile:
    age: int
    sex: str
    diabetes_duration: int
    hba1c: float
    numbness: bool
    tingling: bool
    pain: bool
    sensory_loss: bool
    ulcer_history: bool
    infection_history: bool
    smoking: bool


@dataclass
class RiskAssessmentResult:
    score: int
    level: str
    urgency: str
    factors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score_breakdown: list[str] = field(default_factory=list)


@dataclass
class ImageAnalysisResult:
    modality: str
    summary: str
    alert_level: str
    source_context: str = ""
    sample_id: str = ""
    source_path: str = ""
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class QAResponse:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    matched_nodes: list[str] = field(default_factory=list)
