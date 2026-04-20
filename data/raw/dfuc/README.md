# DFUC 扩展数据目录

当前目录用于放置 DFUC Challenge 或类似糖尿病足溃疡图像数据。

推荐结构：

```text
data/raw/dfuc/
├── images/
├── masks/
├── metadata.csv
└── splits.csv
```

说明：

- `images/`：原始图像
- `masks/`：分割掩膜或标注结果
- `metadata.csv`：可选，记录样本编号、类别、采集来源等
- `splits.csv`：可选，记录训练/验证/测试划分

页面预览逻辑会优先读取：

- `images/` 下的图片作为原图
- `masks/` 下包含 `mask`、`label`、`seg` 等关键词的图片作为掩膜
- 同名样本会自动尝试配对预览

当前项目只把 DFUC 作为扩展演示入口，不作为主线风险评分数据。
