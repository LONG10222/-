from __future__ import annotations

from diabetic_foot_agent.models import PatientProfile, RiskAssessmentResult


def _base_suggestions() -> list[str]:
    return [
        "坚持每日检查足底、趾缝、皮肤颜色和是否有破损。",
        "选择合脚鞋袜，避免赤脚行走，避免局部摩擦和烫伤。",
        "控制血糖并按计划复诊，必要时完善神经病变和血供评估。",
        "中医相关内容定位为护理宣教，可参考足部护理、生活调摄和一般性健康教育建议。",
    ]


def evaluate_risk(profile: PatientProfile) -> RiskAssessmentResult:
    score = 0
    factors: list[str] = []
    score_breakdown: list[str] = []

    def add_points(points: int, reason: str) -> None:
        nonlocal score
        score += points
        score_breakdown.append(f"{reason} (+{points})")

    if profile.age >= 60:
        add_points(1, "年龄 >= 60 岁")
        factors.append("年龄较大，足部并发症风险上升")

    if profile.diabetes_duration >= 10:
        add_points(2, "糖尿病病程 >= 10 年")
        factors.append("糖尿病病程较长")

    if profile.hba1c >= 8.0:
        add_points(2, "HbA1c >= 8.0")
        factors.append("HbA1c 偏高，提示血糖控制欠佳")

    if profile.numbness:
        add_points(1, "存在足麻")
        factors.append("存在足麻症状")

    if profile.tingling:
        add_points(1, "存在刺痛或异常感觉")
        factors.append("存在刺痛或异常感觉")

    if profile.pain:
        add_points(1, "存在足部疼痛")
        factors.append("存在足部疼痛不适")

    if profile.sensory_loss:
        add_points(2, "存在感觉减退")
        factors.append("存在感觉减退，需警惕周围神经病变")

    if profile.ulcer_history:
        add_points(3, "既往足溃疡史")
        factors.append("既往有足溃疡史")

    if profile.infection_history:
        add_points(2, "既往或近期感染史")
        factors.append("既往或近期有感染相关病史")

    if profile.smoking:
        add_points(1, "存在吸烟情况")
        factors.append("吸烟可能增加血供与愈合相关风险")

    if score >= 8 or profile.ulcer_history or profile.infection_history:
        level = "高风险"
        urgency = "建议尽快线下就医或专科评估"
    elif score >= 4:
        level = "中风险"
        urgency = "建议近期门诊评估并加强居家管理"
    else:
        level = "低风险"
        urgency = "建议持续居家监测并按期复诊"

    suggestions = _base_suggestions()
    if profile.numbness or profile.sensory_loss:
        suggestions.append("若存在麻木、感觉减退，建议尽早做足部神经功能相关检查。")
    if profile.ulcer_history or profile.infection_history:
        suggestions.append("有溃疡或感染相关病史者应提高随访频率，不建议自行处理可疑破损。")
    if profile.hba1c >= 8.0:
        suggestions.append("血糖控制是降低足部风险的重要环节，建议与医生沟通降糖目标。")
    if profile.ulcer_history or profile.infection_history:
        suggestions.append("如存在破损、溃疡、感染或明显红肿热痛，不建议自行局部按压或推拿。")

    return RiskAssessmentResult(
        score=score,
        level=level,
        urgency=urgency,
        factors=factors,
        suggestions=suggestions,
        score_breakdown=score_breakdown,
    )
