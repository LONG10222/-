from __future__ import annotations

from datetime import date

from diabetic_foot_agent.models import ImageAnalysisResult, PatientProfile, QAResponse, RiskAssessmentResult


def build_markdown_report(
    profile: PatientProfile | None,
    risk_result: RiskAssessmentResult | None,
    image_result: ImageAnalysisResult | None,
    qa_result: QAResponse | None,
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
        lines.extend(
            [
                "## 3. 图像提示",
                f"- 模态类型：{image_result.modality}",
                f"- 提示等级：{image_result.alert_level}",
                f"- 结论摘要：{image_result.summary}",
                "- 图像发现：",
            ]
        )
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

    lines.extend(
        [
            "## 5. 总结",
            "- 建议将问卷结果、图像提示和线下体检结果联合判断。",
            "- 本报告不进行糖尿病足临床确诊，仅提供高危因素解释、就医建议和护理宣教参考。",
            "- 如存在破损、溃疡、感染、明显红肿或快速加重的不适，应及时线下就医。",
        ]
    )

    return "\n".join(lines)
