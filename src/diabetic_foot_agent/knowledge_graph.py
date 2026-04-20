from __future__ import annotations

import json
from pathlib import Path

from diabetic_foot_agent.models import QAResponse, RiskAssessmentResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_FILE = PROJECT_ROOT / "data" / "schema" / "knowledge_graph_seed.json"


def load_seed_graph() -> list[dict[str, str]]:
    with SEED_FILE.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _match_keywords(query: str) -> list[str]:
    mapping = {
        "麻木": ["麻", "麻木", "感觉减退"],
        "刺痛": ["刺痛", "针刺感", "疼"],
        "就医": ["医院", "就医", "门诊", "转诊"],
        "护理": ["护理", "检查", "自查", "每天", "日常"],
        "中医护理宣教": ["中医", "推拿", "穴位", "按压", "外治", "宣教"],
        "溃疡": ["溃疡", "破损", "伤口"],
    }

    matched = []
    for node, keywords in mapping.items():
        if any(keyword in query for keyword in keywords):
            matched.append(node)
    return matched


def answer_question(query: str, risk_result: RiskAssessmentResult | None = None) -> QAResponse:
    graph = load_seed_graph()
    matched_nodes = _match_keywords(query)

    matched_edges = [
        edge
        for edge in graph
        if any(
            node in edge["subject"] or node in edge["object"]
            for node in matched_nodes
        )
    ]

    answer_parts: list[str] = []

    if "麻木" in matched_nodes or "刺痛" in matched_nodes:
        answer_parts.append("足麻、刺痛和感觉减退常提示周围神经病变风险，建议不要只观察症状变化。")
    if "就医" in matched_nodes or "溃疡" in matched_nodes:
        answer_parts.append("如出现破损、溃疡、明显红肿、渗液或感染表现，应尽快线下就医。")
    if "护理" in matched_nodes:
        answer_parts.append("日常建议坚持足底自查、保持皮肤清洁干燥、避免赤脚和局部摩擦。")
    if "中医护理宣教" in matched_nodes:
        answer_parts.append("中医相关内容仅作为护理宣教与健康教育参考，可结合足部护理和生活调摄理解。")
        answer_parts.append("如存在破损、溃疡、感染或明显红肿热痛，不建议自行局部按压或推拿。")

    if risk_result is not None:
        if risk_result.level == "高风险":
            answer_parts.append("结合当前问卷结果，你属于高风险分层，更应优先做线下专科评估。")
        elif risk_result.level == "中风险":
            answer_parts.append("结合当前问卷结果，建议在近期门诊评估基础上强化居家足部管理。")

    if not answer_parts:
        answer_parts.append("当前问题可从风险因素、足部护理、就医指征和中医护理宣教四个方向展开。")

    evidence = [edge["evidence"] for edge in matched_edges][:4]
    if not evidence:
        evidence = ["当前基于种子图谱回答，后续建议接入更完整的指南和文献证据库。"]

    return QAResponse(
        question=query,
        answer=" ".join(answer_parts),
        evidence=evidence,
        matched_nodes=matched_nodes,
    )
