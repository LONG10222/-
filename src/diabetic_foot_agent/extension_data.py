from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "label", "annotation", "seg")
MASK_SUFFIX_PATTERNS = (
    "_mask",
    "-mask",
    "_label",
    "-label",
    "_annotation",
    "-annotation",
    "_seg",
    "-seg",
)


@dataclass
class ExtensionDatasetStatus:
    key: str
    name: str
    root_path: Path
    available: bool
    file_count: int
    image_count: int
    mask_count: int
    manifests_found: list[str] = field(default_factory=list)
    example_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DFUCSamplePreview:
    sample_id: str
    image_path: Path
    mask_path: Path | None = None


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _looks_like_mask(path: Path) -> bool:
    lower_path = path.as_posix().lower()
    return any(hint in lower_path for hint in MASK_HINTS)


def _normalize_sample_id(path: Path) -> str:
    sample_id = path.stem.lower()
    for pattern in MASK_SUFFIX_PATTERNS:
        if sample_id.endswith(pattern):
            sample_id = sample_id[: -len(pattern)]
            break
    return sample_id


def _build_status(
    key: str,
    name: str,
    relative_dir: str,
    manifest_candidates: list[str],
    notes: list[str],
) -> ExtensionDatasetStatus:
    root_path = RAW_DATA_DIR / relative_dir
    root_path.mkdir(parents=True, exist_ok=True)

    files = [path for path in root_path.rglob("*") if path.is_file() and path.name.lower() != "readme.md"]
    image_files = [path for path in files if _is_image_file(path)]
    mask_files = [
        path
        for path in image_files
        if _looks_like_mask(path)
    ]
    manifests_found = [name for name in manifest_candidates if (root_path / name).exists()]
    example_files = [path.relative_to(root_path).as_posix() for path in sorted(files)[:5]]
    available = bool(image_files or manifests_found)

    return ExtensionDatasetStatus(
        key=key,
        name=name,
        root_path=root_path,
        available=available,
        file_count=len(files),
        image_count=len(image_files),
        mask_count=len(mask_files),
        manifests_found=manifests_found,
        example_files=example_files,
        notes=notes,
    )


def get_extension_dataset_statuses() -> list[ExtensionDatasetStatus]:
    return [
        _build_status(
            key="dfuc",
            name="DFUC 足溃疡图像扩展",
            relative_dir="dfuc",
            manifest_candidates=["metadata.csv", "splits.csv", "labels.csv"],
            notes=[
                "适合做足溃疡图像分割或异常区域演示，不建议作为当前主线风险评分的数据来源。",
                "数据通常需要按许可协议申请，落盘后建议保留图像目录、掩膜目录和任务划分文件。",
            ],
        ),
        _build_status(
            key="standup",
            name="热成像 / STANDUP 扩展",
            relative_dir="standup",
            manifest_candidates=["metadata.csv", "thermal_metadata.csv", "subjects.csv"],
            notes=[
                "适合做热异常提示、热特征研究或多模态接口演示，不建议一开始接入主线决策。",
                "如同时具备 RGB 与热成像图，建议保持同名样本或单独的受试者映射表。",
            ],
        ),
    ]


def get_dfuc_preview_samples(limit: int | None = 12, root_path: Path | None = None) -> list[DFUCSamplePreview]:
    dataset_root = root_path or (RAW_DATA_DIR / "dfuc")
    dataset_root.mkdir(parents=True, exist_ok=True)

    image_files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and _is_image_file(path)
    ]
    non_mask_images = [path for path in image_files if not _looks_like_mask(path)]
    mask_images = [path for path in image_files if _looks_like_mask(path)]

    mask_index = {_normalize_sample_id(path): path for path in sorted(mask_images)}

    previews: list[DFUCSamplePreview] = []
    for image_path in sorted(non_mask_images):
        sample_id = _normalize_sample_id(image_path)
        previews.append(
            DFUCSamplePreview(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_index.get(sample_id),
            )
        )
        if limit is not None and len(previews) >= limit:
            break

    return previews
