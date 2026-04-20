from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetic_foot_agent import (
    analyze_foot_image,
    answer_question,
    build_markdown_report,
    build_sample_options,
    evaluate_risk,
    get_cohort_summary,
)
from diabetic_foot_agent.models import PatientProfile


FAQ_TEMPLATES = [
    "我脚麻但没溃疡怎么办？",
    "什么时候必须去医院？",
    "平时应该怎么检查足部？",
    "中医护理宣教能提供哪些辅助建议？",
]


st.set_page_config(page_title="糖尿病足风险提示与中医护理宣教智能体", layout="wide")


def _ensure_risk_defaults() -> None:
    defaults = {
        "risk_age": 55,
        "risk_sex": "男",
        "risk_diabetes_duration": 8,
        "risk_hba1c": 7.5,
        "risk_smoking": False,
        "risk_ulcer_history": False,
        "risk_infection_history": False,
        "risk_numbness": False,
        "risk_tingling": False,
        "risk_pain": False,
        "risk_sensory_loss": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_nhanes_sample(sample: dict[str, object]) -> None:
    if pd.notna(sample.get("age")):
        st.session_state["risk_age"] = int(sample["age"])
    if sample.get("sex") in {"男", "女"}:
        st.session_state["risk_sex"] = sample["sex"]
    if pd.notna(sample.get("hba1c")):
        st.session_state["risk_hba1c"] = float(sample["hba1c"])
    if pd.notna(sample.get("smoking")):
        st.session_state["risk_smoking"] = bool(int(sample["smoking"]))


def render_overview() -> None:
    st.title("糖尿病足风险提示与中医护理宣教智能体")
    st.caption("公共数据 + 知识图谱 + Streamlit MVP 原型")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("项目定位")
        st.write(
            "围绕糖尿病患者居家管理与基层风险提示辅助场景，整合结构化临床数据、知识图谱和网页原型，"
            "提供风险问卷、知识问答和综合报告。"
        )
        st.subheader("公开数据来源")
        st.markdown(
            "- NHANES：结构化临床与风险因素\n"
            "- IDF / CDC / ADA：知识证据来源\n"
            "- DFUC：可选图像扩展\n"
            "- STANDUP：热成像接口预留"
        )
    with col2:
        st.subheader("当前 MVP")
        st.markdown(
            "1. 风险问卷评估\n"
            "2. 知识图谱问答\n"
            "3. 综合报告生成\n"
            "4. 可选扩展：足部图像演示"
        )
        st.info("说明：当前版本为课程演示原型，主线聚焦风险提示、知识问答和综合报告。")


def render_risk_page() -> None:
    _ensure_risk_defaults()
    st.header("风险问卷")
    st.caption("当前采用规则驱动的综合风险分层，用于课程演示中的风险提示辅助，不直接做临床确诊。")

    summary = get_cohort_summary()
    st.subheader("NHANES 参考队列")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("糖尿病自报样本", summary["sample_count"])
    metric2.metric("HbA1c 可用样本", summary["hba1c_available"])
    metric3.metric("ABI 可用样本", summary["abi_available"])
    metric4.metric("参考高风险样本", summary["high_reference_risk"])

    sample_options = build_sample_options()
    if sample_options:
        st.caption("可以从 NHANES 样本中快速填充基础字段。当前只自动带入年龄、性别、HbA1c 和吸烟情况。")
        label_to_sample = {label: sample for label, sample in sample_options}
        selected_label = st.selectbox("选择 NHANES 参考样本", ["不使用样本填充"] + list(label_to_sample.keys()))
        if selected_label != "不使用样本填充":
            selected_sample = label_to_sample[selected_label]
            fill_col1, fill_col2 = st.columns([1, 2])
            with fill_col1:
                if st.button("应用该样本到问卷", use_container_width=True):
                    _apply_nhanes_sample(selected_sample)
            with fill_col2:
                st.write(
                    f"样本摘要：年龄 {int(selected_sample['age']) if pd.notna(selected_sample.get('age')) else 'NA'}，"
                    f"HbA1c {selected_sample.get('hba1c', 'NA')}，"
                    f"参考风险 {selected_sample.get('reference_risk_level', 'NA')}"
                )
    else:
        st.warning("尚未生成 NHANES 特征文件，请先运行 `python scripts/prepare_nhanes_features.py`。")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("年龄", min_value=18, max_value=100, step=1, key="risk_age")
            sex = st.selectbox("性别", ["男", "女"], key="risk_sex")
            diabetes_duration = st.number_input("糖尿病病程（年）", min_value=0, max_value=50, step=1, key="risk_diabetes_duration")
        with col2:
            hba1c = st.number_input("HbA1c", min_value=4.0, max_value=20.0, step=0.1, format="%.1f", key="risk_hba1c")
            smoking = st.checkbox("有吸烟情况", key="risk_smoking")
            ulcer_history = st.checkbox("既往有足溃疡史", key="risk_ulcer_history")
        with col3:
            infection_history = st.checkbox("既往或近期有感染史", key="risk_infection_history")
            numbness = st.checkbox("存在足麻", key="risk_numbness")
            tingling = st.checkbox("存在刺痛", key="risk_tingling")
            pain = st.checkbox("存在疼痛", key="risk_pain")
            sensory_loss = st.checkbox("存在感觉减退", key="risk_sensory_loss")

        submitted = st.form_submit_button("生成风险评估")

    if submitted:
        profile = PatientProfile(
            age=int(age),
            sex=sex,
            diabetes_duration=int(diabetes_duration),
            hba1c=float(hba1c),
            numbness=numbness,
            tingling=tingling,
            pain=pain,
            sensory_loss=sensory_loss,
            ulcer_history=ulcer_history,
            infection_history=infection_history,
            smoking=smoking,
        )
        risk_result = evaluate_risk(profile)
        st.session_state["profile"] = profile
        st.session_state["risk_result"] = risk_result

    risk_result = st.session_state.get("risk_result")
    if risk_result:
        st.subheader("评估结果")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("风险等级", risk_result.level)
        metric2.metric("风险得分", risk_result.score)
        metric3.metric("就医建议", risk_result.urgency)

        st.markdown("**主要风险因子**")
        for factor in risk_result.factors or ["暂未识别明显高危因子"]:
            st.write(f"- {factor}")

        with st.expander("查看评分依据"):
            score_df = pd.DataFrame(
                [{"规则项": item.split(" (+")[0], "加分": item.split("(+")[-1].rstrip(")")} for item in risk_result.score_breakdown]
            )
            if score_df.empty:
                st.write("当前未触发规则加分项。")
            else:
                st.dataframe(score_df, hide_index=True, use_container_width=True)

        st.markdown("**辅助管理建议**")
        for item in risk_result.suggestions:
            st.write(f"- {item}")


def render_image_page() -> None:
    st.header("足部图像分析")
    modality = st.selectbox("图像模态", ["RGB", "热成像"])
    uploaded_file = st.file_uploader("上传足部图像", type=["png", "jpg", "jpeg"])

    if st.button("执行图像分析", use_container_width=False):
        if not uploaded_file:
            st.warning("请先上传图像。")
        else:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            result = analyze_foot_image(image, modality)
            st.session_state["image_result"] = result
            st.session_state["image_preview"] = image

    image_result = st.session_state.get("image_result")
    preview = st.session_state.get("image_preview")
    if preview is not None:
        st.image(preview, caption="上传图像预览", width=320)

    if image_result:
        st.subheader("图像提示结果")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(image_result.summary)
            for finding in image_result.findings:
                st.write(f"- {finding}")
        with col2:
            st.metric("提示等级", image_result.alert_level)
        metrics_df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in image_result.metrics.items()]
        )
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)


def render_qa_page() -> None:
    st.header("知识图谱问答")
    st.caption("优先使用固定问题模板完成 MVP，后续再扩展更自由的检索与问答。")
    template_question = st.selectbox("常见问题模板", FAQ_TEMPLATES, index=0)
    custom_question = st.text_input("或输入自定义问题", value="")
    question = custom_question.strip() or template_question
    risk_result = st.session_state.get("risk_result")
    if risk_result is not None:
        st.info("当前问答会结合风险问卷结果给出更贴近场景的解释。")
    if st.button("生成问答结果", use_container_width=False):
        qa_result = answer_question(question, risk_result=risk_result)
        st.session_state["qa_result"] = qa_result

    qa_result = st.session_state.get("qa_result")
    if qa_result:
        st.subheader("回答")
        st.write(qa_result.answer)
        if qa_result.matched_nodes:
            st.write("匹配节点：" + "、".join(qa_result.matched_nodes))
        st.markdown("**证据来源**")
        for item in qa_result.evidence:
            st.write(f"- {item}")
        st.caption("提示：如存在破损、溃疡、感染或明显红肿热痛，应优先线下就医，不建议自行局部处理。")


def render_report_page() -> None:
    st.header("综合报告")
    profile = st.session_state.get("profile")
    risk_result = st.session_state.get("risk_result")
    image_result = st.session_state.get("image_result")
    qa_result = st.session_state.get("qa_result")

    if profile is None and qa_result is None:
        st.info("请先完成风险问卷或知识图谱问答，再生成综合报告。")

    report_md = build_markdown_report(profile, risk_result, image_result, qa_result)
    st.markdown(report_md)
    st.download_button(
        label="下载 Markdown 报告",
        data=report_md.encode("utf-8"),
        file_name="diabetic_foot_report.md",
        mime="text/markdown",
    )


page = st.sidebar.radio(
    "功能导航",
    ["项目总览", "风险问卷", "知识图谱问答", "综合报告", "可选扩展：足部图像演示"],
)

if page == "项目总览":
    render_overview()
elif page == "风险问卷":
    render_risk_page()
elif page == "知识图谱问答":
    render_qa_page()
elif page == "综合报告":
    render_report_page()
else:
    render_image_page()
