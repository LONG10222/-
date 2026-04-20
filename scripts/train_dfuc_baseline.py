from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent.dfuc_model import DFUCModelConfig, train_dfuc_baseline


def main() -> None:
    output_dir = PROJECT_ROOT / "artifacts" / "dfuc_baseline"
    metadata = train_dfuc_baseline(output_dir=output_dir, config=DFUCModelConfig())
    print(f"Saved best checkpoint to {metadata['best_checkpoint_path']}")
    print(f"Saved last checkpoint to {metadata['last_checkpoint_path']}")
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Validation samples: {metadata['validation_samples']}")
    print(f"Best val loss: {metadata['best_val_loss']}")
    print(f"Loss history: {metadata['loss_history']}")


if __name__ == "__main__":
    main()
