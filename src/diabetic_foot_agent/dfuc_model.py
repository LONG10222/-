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
    validation_split: float = 0.2
    random_seed: int = 42
    bce_weight: float = 0.5
    dice_weight: float = 0.5


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


def _split_rows(rows: list[dict[str, Any]], validation_split: float, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= 1 or validation_split <= 0:
        return rows, []

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(rows) * validation_split)))
    if val_size >= len(rows):
        val_size = len(rows) - 1

    val_indices = set(indices[:val_size].tolist())
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train_rows, val_rows


def _build_dataset_class(torch: Any, Dataset: Any) -> Any:
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

    return DFUCDataset


def _evaluate_loss(model: Any, dataloader: Any, criterion: Any, torch: Any) -> float:
    if dataloader is None:
        return 0.0

    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item())
            batch_count += 1
    return total_loss / max(batch_count, 1)


def _dice_loss(logits: Any, targets: Any, torch: Any, eps: float = 1e-6) -> Any:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def _combined_segmentation_loss(logits: Any, targets: Any, criterion: Any, torch: Any, config: DFUCModelConfig) -> Any:
    bce = criterion(logits, targets)
    dice = _dice_loss(logits, targets, torch)
    return config.bce_weight * bce + config.dice_weight * dice


def _compute_segmentation_metrics(logits: Any, targets: Any, torch: Any, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean().item()
    foreground_ratio = targets.mean().item()
    return {
        "iou": float(iou),
        "dice": float(dice),
        "foreground_ratio": float(foreground_ratio),
    }


def _evaluate_segmentation(model: Any, dataloader: Any, criterion: Any, torch: Any, config: DFUCModelConfig) -> dict[str, float]:
    if dataloader is None:
        return {"loss": 0.0, "iou": 0.0, "dice": 0.0}

    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    batch_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            logits = model(images)
            loss = _combined_segmentation_loss(logits, masks, criterion, torch, config)
            metrics = _compute_segmentation_metrics(logits, masks, torch)
            total_loss += float(loss.item())
            total_iou += metrics["iou"]
            total_dice += metrics["dice"]
            batch_count += 1

    return {
        "loss": total_loss / max(batch_count, 1),
        "iou": total_iou / max(batch_count, 1),
        "dice": total_dice / max(batch_count, 1),
    }


def train_dfuc_baseline(
    output_dir: Path,
    config: DFUCModelConfig | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    torch, nn, F, DataLoader, Dataset = _require_torch()
    config = config or DFUCModelConfig()
    dataset_root, rows = _load_training_rows(index_path=index_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    DFUCDataset = _build_dataset_class(torch, Dataset)
    train_rows, val_rows = _split_rows(rows, config.validation_split, config.random_seed)
    train_dataset = DFUCDataset(dataset_root, train_rows, config.image_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_rows:
        val_dataset = DFUCDataset(dataset_root, val_rows, config.image_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = _build_unet(nn, F)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoints_dir / "last.pt"
    best_path = checkpoints_dir / "best.pt"
    best_val_loss: float | None = None

    model.train()
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        batch_count = 0
        for images, masks in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = _combined_segmentation_loss(logits, masks, criterion, torch, config)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            metrics = _compute_segmentation_metrics(logits.detach(), masks, torch)
            train_iou += metrics["iou"]
            train_dice += metrics["dice"]
            batch_count += 1
        train_loss = train_loss / max(batch_count, 1)
        train_iou = train_iou / max(batch_count, 1)
        train_dice = train_dice / max(batch_count, 1)
        val_metrics = _evaluate_segmentation(model, val_loader, criterion, torch, config) if val_loader is not None else {
            "loss": train_loss,
            "iou": train_iou,
            "dice": train_dice,
        }
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
            }
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": asdict(config),
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }
        torch.save(checkpoint, last_path)
        if best_val_loss is None or val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, best_path)

    metadata_path = output_dir / "dfuc_baseline.json"
    metadata = {
        "config": asdict(config),
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "loss_history": history,
        "last_checkpoint_path": str(last_path),
        "best_checkpoint_path": str(best_path),
        "best_val_loss": best_val_loss,
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
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
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


def find_dfuc_checkpoint(output_dir: Path | None = None, checkpoint_name: str = "best.pt") -> Path | None:
    artifacts_dir = output_dir or (Path("artifacts") / "dfuc_baseline")
    checkpoint_path = artifacts_dir / "checkpoints" / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def load_dfuc_training_metadata(output_dir: Path | None = None) -> dict[str, Any] | None:
    artifacts_dir = output_dir or (Path("artifacts") / "dfuc_baseline")
    metadata_path = artifacts_dir / "dfuc_baseline.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))
