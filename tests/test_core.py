from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent.knowledge_graph import answer_question
from diabetic_foot_agent.models import PatientProfile
from diabetic_foot_agent.risk_assessment import evaluate_risk


def test_high_risk_patient_is_flagged() -> None:
    profile = PatientProfile(
        age=68,
        sex="男",
        diabetes_duration=15,
        hba1c=9.2,
        numbness=True,
        tingling=True,
        pain=False,
        sensory_loss=True,
        ulcer_history=False,
        infection_history=True,
        smoking=True,
    )

    result = evaluate_risk(profile)

    assert result.level == "高风险"
    assert "线下就医" in result.urgency
    assert result.score >= 8


def test_question_answer_contains_evidence() -> None:
    qa = answer_question("什么时候必须去医院？")

    assert qa.answer
    assert qa.evidence
