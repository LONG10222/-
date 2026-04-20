from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent.dfuc_model import predict_dfuc_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFUC baseline inference on a single image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("weights_path", type=Path)
    parser.add_argument("--output-mask", dest="output_mask", type=Path, default=PROJECT_ROOT / "artifacts" / "dfuc_baseline" / "prediction_mask.png")
    args = parser.parse_args()

    result = predict_dfuc_mask(args.image_path, args.weights_path, args.output_mask)
    print(f"Saved mask to {result['output_mask_path']}")
    print(f"Max probability: {result['max_probability']:.4f}")
    print(f"Mean probability: {result['mean_probability']:.4f}")


if __name__ == "__main__":
    main()
