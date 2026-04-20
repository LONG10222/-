from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent.dfuc_reference import build_dfuc_index, get_dfuc_sample_options, get_dfuc_summary
from diabetic_foot_agent.image_analysis import analyze_foot_image
from diabetic_foot_agent.knowledge_graph import answer_question
from diabetic_foot_agent.models import PatientProfile
from diabetic_foot_agent.extension_data import get_dfuc_preview_samples, get_extension_dataset_statuses
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


def test_extension_dataset_statuses_are_exposed() -> None:
    statuses = get_extension_dataset_statuses()
    keys = {status.key for status in statuses}

    assert {"dfuc", "standup"} <= keys
    for status in statuses:
        assert status.root_path.exists()
        assert status.file_count >= 0


def test_dfuc_preview_samples_pair_images_and_masks(tmp_path) -> None:
    dfuc_root = tmp_path / "dfuc"
    images_dir = dfuc_root / "images"
    masks_dir = dfuc_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    (images_dir / "case_001.jpg").write_bytes(b"fake-image")
    (masks_dir / "case_001_mask.png").write_bytes(b"fake-mask")
    (images_dir / "case_002.jpg").write_bytes(b"fake-image-2")

    previews = get_dfuc_preview_samples(limit=5, root_path=dfuc_root)

    assert len(previews) == 2
    assert previews[0].sample_id == "case_001"
    assert previews[0].mask_path is not None
    assert previews[1].sample_id == "case_002"
    assert previews[1].mask_path is None


def test_dfuc_index_builds_summary_and_options(tmp_path) -> None:
    dfuc_root = tmp_path / "dfuc"
    images_dir = dfuc_root / "images"
    masks_dir = dfuc_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    (images_dir / "case_010.jpg").write_bytes(b"img")
    (masks_dir / "case_010_mask.png").write_bytes(b"mask")
    (images_dir / "case_011.jpg").write_bytes(b"img2")

    df = build_dfuc_index(root_path=dfuc_root)
    assert list(df.columns) == ["sample_id", "image_path", "mask_path", "has_mask"]
    assert len(df) == 2
    assert df["has_mask"].sum() == 1

    summary = {
        "sample_count": int(len(df)),
        "paired_count": int(df["has_mask"].sum()),
        "unpaired_count": int(len(df) - df["has_mask"].sum()),
    }
    assert summary == {"sample_count": 2, "paired_count": 1, "unpaired_count": 1}


def test_image_analysis_keeps_source_metadata() -> None:
    image = Image.new("RGB", (8, 8), color=(200, 20, 20))
    result = analyze_foot_image(
        image,
        "RGB",
        source_context="DFUC 本地样本演示",
        sample_id="case_001",
        source_path="images/case_001.jpg",
    )

    assert result.source_context == "DFUC 本地样本演示"
    assert result.sample_id == "case_001"
    assert result.source_path == "images/case_001.jpg"
