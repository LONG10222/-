from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from diabetic_foot_agent.extension_data import get_dfuc_preview_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DFUC_ROOT = PROJECT_ROOT / "data" / "raw" / "dfuc"
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "dfuc_image_index.csv"


def build_dfuc_index(root_path: Path | None = None) -> pd.DataFrame:
    dataset_root = root_path or DFUC_ROOT
    dataset_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sample in get_dfuc_preview_samples(limit=None, root_path=dataset_root):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path.relative_to(dataset_root).as_posix(),
                "mask_path": sample.mask_path.relative_to(dataset_root).as_posix() if sample.mask_path else "",
                "has_mask": sample.mask_path is not None,
            }
        )

    return pd.DataFrame(rows, columns=["sample_id", "image_path", "mask_path", "has_mask"])


def save_dfuc_index(root_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    dataset_root = root_path or DFUC_ROOT
    index_path = output_path or INDEX_FILE
    index_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dfuc_index(root_path=dataset_root)
    df.to_csv(index_path, index=False)
    load_dfuc_index.cache_clear()
    return df


@lru_cache(maxsize=1)
def load_dfuc_index() -> pd.DataFrame:
    if INDEX_FILE.exists():
        return pd.read_csv(INDEX_FILE)
    return build_dfuc_index()


def get_dfuc_summary() -> dict[str, int]:
    df = load_dfuc_index()
    if df.empty:
        return {"sample_count": 0, "paired_count": 0, "unpaired_count": 0}

    paired_count = int(df["has_mask"].fillna(False).astype(bool).sum())
    return {
        "sample_count": int(len(df)),
        "paired_count": paired_count,
        "unpaired_count": int(len(df) - paired_count),
    }


def get_dfuc_sample_options(limit: int = 200) -> list[tuple[str, dict[str, object]]]:
    df = load_dfuc_index()
    if df.empty:
        return []

    options: list[tuple[str, dict[str, object]]] = []
    for _, row in df.head(limit).iterrows():
        label = f"{row['sample_id']} | {'带掩膜' if bool(row['has_mask']) else '无掩膜'}"
        options.append((label, row.to_dict()))
    return options
