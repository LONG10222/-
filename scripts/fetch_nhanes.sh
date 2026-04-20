#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/raw/nhanes"
mkdir -p "$OUT_DIR"

BASE_URL="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles"
FILES=(
  "DEMO_C.XPT"
  "DIQ_C.XPT"
  "L10_C.XPT"
  "LEXAB_C.XPT"
  "LEXPN_C.XPT"
  "SMQ_C.XPT"
)

for file in "${FILES[@]}"; do
  echo "Downloading $file ..."
  curl -L "$BASE_URL/$file" -o "$OUT_DIR/$file"
done

echo "Done. Files saved in $OUT_DIR"

