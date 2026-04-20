# 数据目录说明

当前仓库已附带主线所需的 NHANES 原始数据与处理后文件，便于直接运行；DFUC、STANDUP 等扩展数据仍需后续单独申请或下载。

当前项目强调的是“多源异构数据资源组织”，而不是把不同来源强行拼成一个患者级统一总表。

## 推荐放置方式

- `raw/nhanes/`：原始 NHANES 数据文件
- `raw/dfuc/`：糖尿病足图像与标注
- `raw/standup/`：RGB 与热成像数据
- `processed/`：统一清洗后的结构化表和图像元数据
- `schema/`：字段 schema 与图谱模板
- `data_cards.md`：各数据源的数据卡
- `schema/field_mapping.csv`：统一字段映射表

## 数据接入建议

1. 主线优先接入 NHANES 结构化数据，先完成风险提示原型。
2. 图像数据按任务单独处理，不与 NHANES 做 patient-level 直接融合。
3. 指南/文献知识抽取为实体、关系、证据三类表。
4. 结构化数据、图像元数据和知识证据通过统一 schema 组织，但保留各自任务边界。

## 当前落地文件

- `data_cards.md`：记录每个数据源的来源、任务、粒度、限制和当前用途。
- `schema/dataset_schema.yaml`：定义统一字段 schema。
- `schema/field_mapping.csv`：定义统一字段与来源字段的映射关系。
- `schema/knowledge_graph_seed.json`：提供知识图谱种子三元组。
