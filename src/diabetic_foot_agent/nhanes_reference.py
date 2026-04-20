from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "nhanes_risk_features.csv"


@lru_cache(maxsize=1)
def load_nhanes_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_FILE)


def get_cohort_summary() -> dict[str, int]:
    df = load_nhanes_features()
    if df.empty:
        return {
            "sample_count": 0,
            "hba1c_available": 0,
            "abi_available": 0,
            "high_reference_risk": 0,
        }

    return {
        "sample_count": int(len(df)),
        "hba1c_available": int(df["hba1c"].notna().sum()),
        "abi_available": int(df["min_abi"].notna().sum()),
        "high_reference_risk": int((df["reference_risk_level"] == "高风险").sum()),
    }


def build_sample_options(limit: int = 200) -> list[tuple[str, dict[str, object]]]:
    df = load_nhanes_features()
    if df.empty:
        return []

    display_df = df.sort_values(
        by=["reference_score", "hba1c", "age"],
        ascending=[False, False, False],
        na_position="last",
    ).head(limit)

    options: list[tuple[str, dict[str, object]]] = []
    for _, row in display_df.iterrows():
        label = (
            f"{int(row['patient_id'])} | {row.get('sex', '未知')} | "
            f"{int(row['age']) if pd.notna(row['age']) else 'NA'} 岁 | "
            f"HbA1c {row['hba1c'] if pd.notna(row['hba1c']) else 'NA'} | "
            f"参考风险 {row['reference_risk_level']}"
        )
        options.append((label, row.to_dict()))
    return options

