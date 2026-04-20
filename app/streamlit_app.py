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
    find_dfuc_checkpoint,
    get_dfuc_sample_options,
    get_dfuc_summary,
    build_sample_options,
    evaluate_risk,
    get_dfuc_preview_samples,
    get_extension_dataset_statuses,
    get_cohort_summary,
    load_dfuc_training_metadata,
    predict_dfuc_mask,
    save_dfuc_index,
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
            "- DFUC：首选图像扩展路线\n"
            "- IDF / CDC / ADA：知识证据来源\n"
            "- STANDUP：热成像扩展说明"
        )
    with col2:
        st.subheader("当前 MVP")
        st.markdown(
            "1. 风险问卷评估\n"
            "2. 知识图谱问答\n"
            "3. 综合报告生成\n"
            "4. 可选扩展：图像 / DFUC / 热成像入口"
        )
        st.info("说明：当前版本为课程演示原型，主线聚焦风险提示、知识问答和综合报告。")


def _render_extension_dataset_status(dataset_key: str) -> None:
    status = {item.key: item for item in get_extension_dataset_statuses()}[dataset_key]

    st.subheader(status.name)
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("本地状态", "已接入" if status.available else "未接入")
    metric2.metric("文件数", status.file_count)
    metric3.metric("图像数", status.image_count)
    metric4.metric("掩膜/标注图像", status.mask_count)

    st.caption(f"目录位置：{status.root_path}")

    if status.manifests_found:
        st.write("检测到的元数据文件：" + "、".join(status.manifests_found))
    else:
        st.write("尚未检测到元数据文件，可后续补充 `metadata.csv`、`splits.csv` 或类似清单。")

    if status.example_files:
        st.markdown("**已发现示例文件**")
        for item in status.example_files:
            st.write(f"- {item}")
    else:
        st.warning("当前目录只有说明文件，尚未检测到可用图像或元数据。")

    st.markdown("**接入说明**")
    for note in status.notes:
        st.write(f"- {note}")


def _render_dfuc_sample_preview() -> None:
    preview_limit = st.slider("预览样本数", min_value=3, max_value=12, value=6, step=3)
    previews = get_dfuc_preview_samples(limit=preview_limit)

    if not previews:
        st.info("当前尚未检测到 DFUC 图像样本。把图片放入 `data/raw/dfuc/images/` 后，这里会自动显示预览。")
        return

    st.markdown("**样本预览**")
    for index in range(0, len(previews), 3):
        columns = st.columns(3)
        for column, sample in zip(columns, previews[index:index + 3]):
            with column:
                st.image(str(sample.image_path), caption=f"{sample.sample_id} | 原图", use_container_width=True)
                if sample.mask_path is not None:
                    st.image(str(sample.mask_path), caption=f"{sample.sample_id} | 掩膜", use_container_width=True)
                else:
                    st.caption("未检测到对应掩膜")


def _render_dfuc_workspace() -> None:
    summary = get_dfuc_summary()
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("DFUC 样本数", summary["sample_count"])
    metric2.metric("已配对掩膜", summary["paired_count"])
    metric3.metric("未配对原图", summary["unpaired_count"])

    training_summary = load_dfuc_training_metadata(PROJECT_ROOT / "artifacts" / "dfuc_baseline")
    if training_summary is not None:
        st.markdown("**训练面板**")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("训练样本", training_summary.get("train_samples", "NA"))
        metric2.metric("验证样本", training_summary.get("validation_samples", "NA"))
        metric3.metric("最佳验证损失", f"{training_summary.get('best_val_loss', 0):.4f}" if training_summary.get("best_val_loss") is not None else "NA")

        loss_history = training_summary.get("loss_history") or []
        if loss_history:
            history_df = pd.DataFrame(loss_history)
            st.line_chart(history_df.set_index("epoch")[["train_loss", "val_loss"]], use_container_width=True)
            st.line_chart(history_df.set_index("epoch")[["train_iou", "val_iou", "train_dice", "val_dice"]], use_container_width=True)
            st.dataframe(history_df, hide_index=True, use_container_width=True)

    if st.button("刷新 DFUC 索引", use_container_width=False):
        refreshed = save_dfuc_index()
        st.success(f"已刷新 DFUC 索引，共 {len(refreshed)} 条样本。")

    sample_options = get_dfuc_sample_options()
    if not sample_options:
        st.info("当前还没有可索引的 DFUC 图像。请先把图片放入 `data/raw/dfuc/images/`。")
        return

    label_to_sample = {label: sample for label, sample in sample_options}
    selected_label = st.selectbox("选择 DFUC 样本", list(label_to_sample.keys()))
    selected_sample = label_to_sample[selected_label]

    dfuc_root = PROJECT_ROOT / "data" / "raw" / "dfuc"
    image_path = dfuc_root / str(selected_sample["image_path"])
    mask_value = str(selected_sample.get("mask_path", "")).strip()
    mask_path = dfuc_root / mask_value if mask_value else None

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(str(image_path), caption=f"{selected_sample['sample_id']} | 原图", use_container_width=True)
    with preview_col2:
        if mask_path and mask_path.exists():
            st.image(str(mask_path), caption=f"{selected_sample['sample_id']} | 掩膜", use_container_width=True)
        else:
            st.info("当前样本未检测到对应掩膜。")

    st.caption(
        f"原图路径：{selected_sample['image_path']} | "
        f"掩膜：{selected_sample['mask_path'] or '无'}"
    )

    if st.button("使用该 DFUC 样本执行演示分析", use_container_width=False):
        image = Image.open(image_path)
        result = analyze_foot_image(
            image,
            "RGB",
            source_context="DFUC 本地样本演示",
            sample_id=str(selected_sample["sample_id"]),
            source_path=str(selected_sample["image_path"]),
        )
        st.session_state["image_result"] = result
        st.session_state["image_preview"] = image
        st.success("已将所选 DFUC 样本载入图像演示分析。")

    checkpoint_path = find_dfuc_checkpoint(PROJECT_ROOT / "artifacts" / "dfuc_baseline")
    if checkpoint_path is not None:
        st.caption(f"检测到本地 checkpoint：{checkpoint_path}")
        if st.button("使用本地 DFUC checkpoint 执行分割推理", use_container_width=False):
            try:
                output_mask = PROJECT_ROOT / "artifacts" / "dfuc_baseline" / f"{selected_sample['sample_id']}_pred_mask.png"
                inference_result = predict_dfuc_mask(image_path, checkpoint_path, output_mask)
                st.session_state["dfuc_pred_mask_path"] = inference_result["output_mask_path"]
                st.session_state["dfuc_pred_stats"] = inference_result
                st.session_state["dfuc_selected_sample"] = selected_sample
                st.success("DFUC 本地推理已完成。")
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"DFUC 推理失败：{exc}")
    else:
        st.info("当前未检测到本地 DFUC checkpoint。先执行训练脚本后，这里会开放分割推理按钮。")

    pred_mask_path = st.session_state.get("dfuc_pred_mask_path")
    pred_stats = st.session_state.get("dfuc_pred_stats")
    if pred_mask_path:
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.image(pred_mask_path, caption="预测掩膜", use_container_width=True)
        with pred_col2:
            if pred_stats:
                st.metric("最大概率", f"{pred_stats['max_probability']:.4f}")
                st.metric("平均概率", f"{pred_stats['mean_probability']:.4f}")


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
    st.header("扩展入口：图像 / DFUC / 热成像")
    st.caption("扩展页用于图像演示和后续数据接入，不替代主线风险评分。")

    tab_demo, tab_dfuc, tab_thermal = st.tabs(["图像演示", "DFUC 数据入口", "热成像入口"])

    with tab_demo:
        source_context = st.selectbox(
            "演示场景",
            ["居家 RGB 拍照演示", "DFUC 风格溃疡图像演示", "热成像研究接口演示"],
        )
        modality = st.selectbox("图像模态", ["RGB", "热成像"])
        uploaded_file = st.file_uploader("上传足部图像", type=["png", "jpg", "jpeg"])

        if source_context == "DFUC 风格溃疡图像演示":
            st.info("当前仅做启发式异常提示，不做真实分割推理；若后续接入 DFUC，可在此页替换为分割模型。")
        elif source_context == "热成像研究接口演示":
            st.info("当前热成像逻辑仅基于亮度统计与热点差值做演示，后续可替换为真实热特征流程。")

        if st.button("执行图像分析", use_container_width=False):
            if not uploaded_file:
                st.warning("请先上传图像。")
            else:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                result = analyze_foot_image(
                    image,
                    modality,
                    source_context=source_context,
                    sample_id="manual_upload",
                    source_path=uploaded_file.name,
                )
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
                if image_result.source_context:
                    st.write(f"来源场景：{image_result.source_context}")
                if image_result.sample_id and image_result.sample_id != "manual_upload":
                    st.write(f"样本编号：{image_result.sample_id}")
                if image_result.source_path:
                    st.caption(f"来源路径：{image_result.source_path}")
                st.write(image_result.summary)
                for finding in image_result.findings:
                    st.write(f"- {finding}")
            with col2:
                st.metric("提示等级", image_result.alert_level)
            metrics_df = pd.DataFrame(
                [{"metric": key, "value": value} for key, value in image_result.metrics.items()]
            )
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with tab_dfuc:
        _render_extension_dataset_status("dfuc")
        st.markdown("**DFUC 首选接入路线**")
        st.write("当前图像扩展优先围绕 DFUC 本地数据目录组织和样本演示展开；热成像继续保留为次级扩展入口。")
        st.code(
            "data/raw/dfuc/\n"
            "  images/\n"
            "  masks/\n"
            "  metadata.csv\n"
            "  splits.csv",
            language="text",
        )
        st.caption("如后续拿到 DFUC 许可数据，优先保持图像与掩膜目录分离，并补一份样本划分文件。")
        st.markdown("**最小模型入口**")
        st.code(
            "pip install .[vision]\n"
            "python scripts/prepare_dfuc_index.py\n"
            "python scripts/train_dfuc_baseline.py\n"
            "python scripts/predict_dfuc_baseline.py <image_path> <weights_path>",
            language="bash",
        )
        _render_dfuc_workspace()
        _render_dfuc_sample_preview()

    with tab_thermal:
        _render_extension_dataset_status("standup")
        st.info("热成像目前保留为扩展入口，不参与当前 DFUC 优先路线下的主图像演示流程。")
        st.code(
            "data/raw/standup/\n"
            "  rgb/\n"
            "  thermal/\n"
            "  metadata.csv\n"
            "  subjects.csv",
            language="text",
        )
        st.caption("如后续接入热成像设备或研究数据，建议统一样本命名，便于 RGB 与热图做一一映射。")


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
    dfuc_training_summary = load_dfuc_training_metadata(PROJECT_ROOT / "artifacts" / "dfuc_baseline")
    dfuc_pred_stats = st.session_state.get("dfuc_pred_stats")
    dfuc_pred_mask_path = st.session_state.get("dfuc_pred_mask_path")

    if (
        profile is None
        and risk_result is None
        and image_result is None
        and qa_result is None
        and dfuc_training_summary is None
        and dfuc_pred_stats is None
    ):
        st.info("请先完成风险问卷或知识图谱问答，再生成综合报告。")
        return

    report_md = build_markdown_report(
        profile,
        risk_result,
        image_result,
        qa_result,
        dfuc_training_summary=dfuc_training_summary,
        dfuc_inference_summary=dfuc_pred_stats,
    )
    st.markdown(report_md)

    if dfuc_training_summary is not None:
        st.subheader("DFUC 训练结果")
        metric1, metric2 = st.columns(2)
        metric1.metric("训练样本", dfuc_training_summary.get("train_samples", "NA"))
        metric2.metric("验证样本", dfuc_training_summary.get("validation_samples", "NA"))
        loss_history = dfuc_training_summary.get("loss_history") or []
        if loss_history:
            st.dataframe(pd.DataFrame(loss_history), hide_index=True, use_container_width=True)

    if dfuc_pred_stats is not None:
        st.subheader("DFUC 推理结果")
        metric1, metric2 = st.columns(2)
        metric1.metric("最大概率", f"{dfuc_pred_stats['max_probability']:.4f}")
        metric2.metric("平均概率", f"{dfuc_pred_stats['mean_probability']:.4f}")
        if dfuc_pred_mask_path:
            st.image(dfuc_pred_mask_path, caption="DFUC 预测掩膜", use_container_width=False)

    st.download_button(
        label="下载 Markdown 报告",
        data=report_md.encode("utf-8"),
        file_name="diabetic_foot_report.md",
        mime="text/markdown",
    )


page = st.sidebar.radio(
    "功能导航",
    ["项目总览", "风险问卷", "知识图谱问答", "综合报告", "可选扩展：图像与热成像入口"],
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
