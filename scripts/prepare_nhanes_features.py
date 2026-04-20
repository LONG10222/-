from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_risk_base.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "nhanes_risk_features.csv"


def build_reference_risk(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat = feat[feat["diabetes_self_report"] == 1].copy()
    feat["hba1c_high_flag"] = (feat["hba1c"] >= 8.0).astype("Int64")
    feat["older_age_flag"] = (feat["age"] >= 60).astype("Int64")
    feat["smoking_flag"] = feat["smoking"].fillna(0).astype("Int64")
    feat["abi_risk_flag"] = feat["abnormal_abi_flag"].fillna(0).astype("Int64")
    feat["lesion_risk_flag"] = feat["lesion_any_flag"].fillna(0).astype("Int64")

    feat["reference_score"] = (
        feat["older_age_flag"].fillna(0)
        + feat["hba1c_high_flag"].fillna(0) * 2
        + feat["smoking_flag"].fillna(0)
        + feat["abi_risk_flag"].fillna(0) * 2
        + feat["lesion_risk_flag"].fillna(0) * 3
    )

    feat["reference_risk_level"] = "低风险"
    feat.loc[feat["reference_score"] >= 3, "reference_risk_level"] = "中风险"
    feat.loc[feat["reference_score"] >= 6, "reference_risk_level"] = "高风险"
    feat["reference_note"] = "基于 NHANES 可获得字段生成的参考分层，仅用于风险提示原型。"

    columns = [
        "patient_id",
        "age",
        "sex",
        "hba1c",
        "smoking",
        "min_abi",
        "abnormal_abi_flag",
        "lesion_any_flag",
        "older_age_flag",
        "hba1c_high_flag",
        "smoking_flag",
        "abi_risk_flag",
        "lesion_risk_flag",
        "reference_score",
        "reference_risk_level",
        "reference_note",
    ]
    return feat[columns].copy()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    features = build_reference_risk(df)
    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")
    print(f"Rows: {len(features)}, Cols: {len(features.columns)}")


if __name__ == "__main__":
    main()

