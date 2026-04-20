from __future__ import annotations

import numpy as np
from PIL import Image

from diabetic_foot_agent.models import ImageAnalysisResult


def analyze_foot_image(
    image: Image.Image,
    modality: str,
    source_context: str = "",
    sample_id: str = "",
    source_path: str = "",
) -> ImageAnalysisResult:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    brightness = rgb.mean(axis=2)

    redness_mask = (
        (rgb[:, :, 0] > 140)
        & (rgb[:, :, 0] > rgb[:, :, 1] * 1.15)
        & (rgb[:, :, 0] > rgb[:, :, 2] * 1.15)
    )
    redness_ratio = float(redness_mask.mean())
    contrast = float(np.std(brightness))

    findings: list[str] = []
    alert_score = 0

    if redness_ratio > 0.10:
        findings.append("图像中偏红区域占比偏高，建议结合肉眼检查是否存在红肿或摩擦点。")
        alert_score += 1

    if contrast > 45:
        findings.append("明暗差异较大，建议关注是否存在颜色异常、结痂或破损边界。")
        alert_score += 1

    metrics = {
        "redness_ratio": round(redness_ratio, 4),
        "contrast": round(contrast, 2),
    }

    if modality == "热成像":
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        hotspot_delta = float(np.percentile(gray, 95) - np.percentile(gray, 50))
        metrics["hotspot_delta"] = round(hotspot_delta, 2)
        if hotspot_delta > 35:
            findings.append("热成像相对热点较明显，可作为进一步检查的提示信号。")
            alert_score += 2
    else:
        dark_ratio = float((brightness < 45).mean())
        metrics["dark_ratio"] = round(dark_ratio, 4)
        if dark_ratio > 0.12:
            findings.append("暗色区域占比较多，如伴随破损或渗出应尽快线下就医。")
            alert_score += 1

    if not findings:
        findings.append("未见明显异常图像信号，但仍建议结合症状和线下检查综合判断。")

    if alert_score >= 3:
        alert_level = "高提示"
        summary = "图像中存在较明显的异常提示，建议尽快做进一步评估。"
    elif alert_score >= 1:
        alert_level = "中提示"
        summary = "图像中存在一定异常征象，建议加强自查并结合问卷结果判断。"
    else:
        alert_level = "低提示"
        summary = "当前图像未见明显高风险征象，建议保持规律足部检查。"

    return ImageAnalysisResult(
        modality=modality,
        summary=summary,
        alert_level=alert_level,
        source_context=source_context,
        sample_id=sample_id,
        source_path=source_path,
        findings=findings,
        metrics=metrics,
    )
