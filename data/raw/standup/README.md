# 热成像 / STANDUP 扩展数据目录

当前目录用于放置足底 RGB 与热成像扩展数据。

推荐结构：

```text
data/raw/standup/
├── rgb/
├── thermal/
├── metadata.csv
└── subjects.csv
```

说明：

- `rgb/`：普通 RGB 图像
- `thermal/`：热成像图像
- `metadata.csv`：可选，记录采集条件、模态类型、样本标签
- `subjects.csv`：可选，记录受试者级映射关系

当前项目将热成像数据作为扩展接口与研究型演示，不直接纳入主线临床风险分层。
