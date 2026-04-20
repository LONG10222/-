from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from diabetic_foot_agent.dfuc_reference import DFUC_ROOT, INDEX_FILE, load_dfuc_index


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError(
            "DFUC 模型入口需要安装可选视觉依赖。请先执行 "
            "`pip install .[vision]` 或单独安装 torch/torchvision。"
        ) from exc
    return torch, nn, F, DataLoader, Dataset


@dataclass
class DFUCModelConfig:
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-3


class _DoubleConv:
    def __init__(self, nn: Any, in_channels: int, out_channels: int) -> None:
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def __call__(self, x: Any) -> Any:
        return self.block(x)


def _build_unet(nn: Any, F: Any) -> Any:
    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = _DoubleConv(nn, 3, 16).block
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = _DoubleConv(nn, 16, 32).block
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv(nn, 32, 64).block
            self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec2 = _DoubleConv(nn, 64, 32).block
            self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.dec1 = _DoubleConv(nn, 32, 16).block
            self.head = nn.Conv2d(16, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            x1 = self.enc1(x)
            x2 = self.enc2(self.pool1(x1))
            x3 = self.bottleneck(self.pool2(x2))

            y2 = self.up2(x3)
            if y2.shape[-2:] != x2.shape[-2:]:
                y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            y2 = self.dec2(torch.cat([y2, x2], dim=1))

            y1 = self.up1(y2)
            if y1.shape[-2:] != x1.shape[-2:]:
                y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            y1 = self.dec1(torch.cat([y1, x1], dim=1))
            return self.head(y1)

    torch, *_ = _require_torch()
    return TinyUNet()


def _load_training_rows(index_path: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    dataset_root = DFUC_ROOT
    if index_path and index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = load_dfuc_index()
    if df.empty:
        raise ValueError("DFUC 索引为空。请先准备本地图像并执行 `python scripts/prepare_dfuc_index.py`。")

    df = df[df["has_mask"].fillna(False).astype(bool)].copy()
    if df.empty:
        raise ValueError("DFUC 索引中没有带掩膜的样本，当前无法训练最小分割模型。")

    return dataset_root, df.to_dict(orient="records")


def train_dfuc_baseline(
    output_dir: Path,
    config: DFUCModelConfig | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    torch, nn, F, DataLoader, Dataset = _require_torch()
    config = config or DFUCModelConfig()
    dataset_root, rows = _load_training_rows(index_path=index_path)

    class DFUCDataset(Dataset):
        def __init__(self, root_dir: Path, samples: list[dict[str, Any]], image_size: int) -> None:
            self.root_dir = root_dir
            self.samples = samples
            self.image_size = image_size

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            sample = self.samples[index]
            image = Image.open(self.root_dir / sample["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
            mask = Image.open(self.root_dir / sample["mask_path"]).convert("L").resize((self.image_size, self.image_size))

            image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)
            mask_tensor = (mask_tensor > 0.5).float()
            return image_tensor, mask_tensor

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DFUCDataset(dataset_root, rows, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = _build_unet(nn, F)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, masks in dataloader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        history.append(epoch_loss / max(batch_count, 1))

    weights_path = output_dir / "dfuc_baseline.pt"
    metadata_path = output_dir / "dfuc_baseline.json"
    torch.save(model.state_dict(), weights_path)
    metadata = {
        "config": asdict(config),
        "train_samples": len(dataset),
        "loss_history": history,
        "weights_path": str(weights_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def predict_dfuc_mask(
    image_path: Path,
    weights_path: Path,
    output_mask_path: Path,
    image_size: int = 256,
) -> dict[str, Any]:
    torch, nn, F, *_ = _require_torch()

    model = _build_unet(nn, F)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size, image_size))
    image_tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = Image.fromarray((probs * 255).astype("uint8"), mode="L").resize(image.size)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_mask_path)

    return {
        "image_path": str(image_path),
        "weights_path": str(weights_path),
        "output_mask_path": str(output_mask_path),
        "max_probability": float(probs.max()),
        "mean_probability": float(probs.mean()),
    }
