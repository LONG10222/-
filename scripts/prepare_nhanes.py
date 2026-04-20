from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nhanes"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "nhanes_risk_base.csv"


def _load_xpt(filename: str) -> pd.DataFrame:
    return pd.read_sas(RAW_DIR / filename)


def _clean_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    # NHANES often uses 7/8/9 and repeated digits as non-substantive codes.
    invalid_codes = {
        -1,
        7, 8, 9,
        77, 88, 99,
        777, 888, 999,
        7777, 8888, 9999,
    }
    numeric = numeric.where(~numeric.isin(invalid_codes))

    # Some SAS special missings may appear as extremely small floats after import.
    numeric = numeric.mask(numeric.abs() < 1e-30)
    return numeric


def _yes_no(series: pd.Series) -> pd.Series:
    cleaned = _clean_series(series)
    return cleaned.map({1.0: 1, 2.0: 0})


def build_base_table() -> pd.DataFrame:
    demo = _load_xpt("DEMO_C.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR"]].copy()
    diq = _load_xpt("DIQ_C.XPT")[["SEQN", "DIQ010", "DIQ050", "DIQ070"]].copy()
    l10 = _load_xpt("L10_C.XPT")[["SEQN", "LBXGH"]].copy()
    lexab = _load_xpt("LEXAB_C.XPT")[["SEQN", "LEXLABPI", "LEXRABPI"]].copy()
    lexpn = _load_xpt("LEXPN_C.XPT")[["SEQN", "LEALPN", "LEARPN", "LEALLES", "LEARLES"]].copy()
    smq = _load_xpt("SMQ_C.XPT")[["SEQN", "SMQ020"]].copy()

    merged = demo.merge(diq, on="SEQN", how="left")
    merged = merged.merge(l10, on="SEQN", how="left")
    merged = merged.merge(lexab, on="SEQN", how="left")
    merged = merged.merge(lexpn, on="SEQN", how="left")
    merged = merged.merge(smq, on="SEQN", how="left")

    merged = merged.rename(
        columns={
            "SEQN": "patient_id",
            "RIAGENDR": "sex_code",
            "RIDAGEYR": "age",
            "DIQ010": "diabetes_self_report_code",
            "DIQ050": "insulin_now_code",
            "DIQ070": "diabetes_pills_code",
            "LBXGH": "hba1c",
            "LEXLABPI": "left_abi",
            "LEXRABPI": "right_abi",
            "LEALPN": "left_neuropathy_raw",
            "LEARPN": "right_neuropathy_raw",
            "LEALLES": "left_lesion_raw",
            "LEARLES": "right_lesion_raw",
            "SMQ020": "smoking_100_cigarettes_code",
        }
    )

    merged["sex"] = merged["sex_code"].map({1.0: "男", 2.0: "女"})
    merged["age"] = _clean_series(merged["age"])
    merged["hba1c"] = _clean_series(merged["hba1c"])
    merged["left_abi"] = _clean_series(merged["left_abi"])
    merged["right_abi"] = _clean_series(merged["right_abi"])
    merged["min_abi"] = merged[["left_abi", "right_abi"]].min(axis=1)
    merged["abnormal_abi_flag"] = (merged["min_abi"] < 0.9).astype("Int64")

    merged["diabetes_self_report"] = _yes_no(merged["diabetes_self_report_code"])
    merged["insulin_now"] = _yes_no(merged["insulin_now_code"])
    merged["diabetes_pills_now"] = _yes_no(merged["diabetes_pills_code"])
    merged["smoking"] = _yes_no(merged["smoking_100_cigarettes_code"])

    merged["left_neuropathy_raw"] = _clean_series(merged["left_neuropathy_raw"])
    merged["right_neuropathy_raw"] = _clean_series(merged["right_neuropathy_raw"])
    merged["left_lesion_raw"] = _clean_series(merged["left_lesion_raw"])
    merged["right_lesion_raw"] = _clean_series(merged["right_lesion_raw"])

    merged["lesion_any_flag"] = (
        (merged["left_lesion_raw"] == 1.0) | (merged["right_lesion_raw"] == 1.0)
    ).astype("Int64")

    ordered_columns = [
        "patient_id",
        "age",
        "sex",
        "diabetes_self_report",
        "insulin_now",
        "diabetes_pills_now",
        "hba1c",
        "smoking",
        "left_abi",
        "right_abi",
        "min_abi",
        "abnormal_abi_flag",
        "left_neuropathy_raw",
        "right_neuropathy_raw",
        "left_lesion_raw",
        "right_lesion_raw",
        "lesion_any_flag",
    ]
    return merged[ordered_columns].copy()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base = build_base_table()
    base.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")
    print(f"Rows: {len(base)}, Cols: {len(base.columns)}")


if __name__ == "__main__":
    main()

