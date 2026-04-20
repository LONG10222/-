# NHANES 原始数据说明

当前目录存放的是项目主线用到的 NHANES 2003-2004 周期公开原始文件。

已下载文件：

- `DEMO_C.XPT`
- `DIQ_C.XPT`
- `L10_C.XPT`
- `LEXAB_C.XPT`
- `LEXPN_C.XPT`
- `SMQ_C.XPT`

这些文件都可以通过 `SEQN` 做受试者级关联。

推荐处理顺序：

1. 读取所有 `XPT` 文件
2. 选取主线字段
3. 按 `SEQN` 合并
4. 输出 `processed/nhanes_risk_base.csv`

