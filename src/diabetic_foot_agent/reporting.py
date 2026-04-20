from __future__ import annotations

from datetime import date

from diabetic_foot_agent.models import ImageAnalysisResult, PatientProfile, QAResponse, RiskAssessmentResult


def build_markdown_report(
    profile: PatientProfile | None,
    risk_result: RiskAssessmentResult | None,
    image_result: ImageAnalysisResult | None,
    qa_result: QAResponse | None,
    dfuc_training_summary: dict | None = None,
    dfuc_inference_summary: dict | None = None,
) -> str:
    lines = [
        "# 糖尿病足风险提示与中医护理宣教报告",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        "- 说明：本报告仅用于课程原型演示、风险提示和健康教育，不替代医生诊断。",
        "",
    ]

    if profile is not None:
        lines.extend(
            [
                "## 1. 基本信息",
                f"- 年龄：{profile.age}",
                f"- 性别：{profile.sex}",
                f"- 糖尿病病程：{profile.diabetes_duration} 年",
                f"- HbA1c：{profile.hba1c}",
                "",
            ]
        )

    if risk_result is not None:
        lines.extend(
            [
                "## 2. 风险评估",
                f"- 风险等级：{risk_result.level}",
                f"- 风险得分：{risk_result.score}",
                f"- 就医建议：{risk_result.urgency}",
                "- 主要风险因子：",
            ]
        )
        lines.extend([f"  - {factor}" for factor in risk_result.factors] or ["  - 暂未识别明显高危因子"])
        lines.extend(["- 评分依据："])
        lines.extend([f"  - {item}" for item in risk_result.score_breakdown] or ["  - 当前未触发规则加分项"])
        lines.extend(["- 辅助管理建议："])
        lines.extend([f"  - {suggestion}" for suggestion in risk_result.suggestions])
        lines.append("")

    if image_result is not None:
        image_lines = [
            "## 3. 图像提示",
            f"- 模态类型：{image_result.modality}",
            f"- 提示等级：{image_result.alert_level}",
            f"- 结论摘要：{image_result.summary}",
        ]
        if image_result.source_context:
            image_lines.append(f"- 来源场景：{image_result.source_context}")
        if image_result.sample_id:
            image_lines.append(f"- 样本编号：{image_result.sample_id}")
        if image_result.source_path:
            image_lines.append(f"- 样本路径：{image_result.source_path}")
        image_lines.append("- 图像发现：")
        lines.extend(image_lines)
        lines.extend([f"  - {item}" for item in image_result.findings])
        lines.append("")

    if qa_result is not None:
        lines.extend(
            [
                "## 4. 知识图谱问答摘要",
                f"- 提问：{qa_result.question}",
                f"- 回答：{qa_result.answer}",
                "- 证据来源：",
            ]
        )
        lines.extend([f"  - {item}" for item in qa_result.evidence])
        lines.append("")

    if dfuc_training_summary is not None:
        lines.extend(
            [
                "## 5. DFUC 训练摘要",
                f"- 训练样本数：{dfuc_training_summary.get('train_samples', 'NA')}",
                f"- 验证样本数：{dfuc_training_summary.get('validation_samples', 'NA')}",
                f"- 最佳 checkpoint：{dfuc_training_summary.get('best_checkpoint_path', 'NA')}",
                f"- 最新 checkpoint：{dfuc_training_summary.get('last_checkpoint_path', 'NA')}",
            ]
        )
        loss_history = dfuc_training_summary.get("loss_history") or []
        if loss_history:
            last_epoch = loss_history[-1]
            lines.extend(
                [
                    f"- 最后一轮训练损失：{last_epoch.get('train_loss', 'NA')}",
                    f"- 最后一轮验证损失：{last_epoch.get('val_loss', 'NA')}",
                ]
            )
        lines.append("")

    if dfuc_inference_summary is not None:
        lines.extend(
            [
                "## 6. DFUC 推理摘要",
                f"- 输入图像：{dfuc_inference_summary.get('image_path', 'NA')}",
                f"- 使用权重：{dfuc_inference_summary.get('weights_path', 'NA')}",
                f"- 输出掩膜：{dfuc_inference_summary.get('output_mask_path', 'NA')}",
                f"- 最大概率：{dfuc_inference_summary.get('max_probability', 'NA')}",
                f"- 平均概率：{dfuc_inference_summary.get('mean_probability', 'NA')}",
            ]
        )
        lines.append("")

    lines.extend(
        [
            "## 7. 总结",
            "- 建议将问卷结果、图像提示和线下体检结果联合判断。",
            "- 本报告不进行糖尿病足临床确诊，仅提供高危因素解释、就医建议和护理宣教参考。",
            "- 如存在破损、溃疡、感染、明显红肿或快速加重的不适，应及时线下就医。",
        ]
    )

    return "\n".join(lines)
