from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent.dfuc_reference import INDEX_FILE, save_dfuc_index


def main() -> None:
    df = save_dfuc_index()
    print(f"Saved {INDEX_FILE}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
