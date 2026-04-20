# 数据获取指南

本指南分成两部分：

- 已经帮你下载到本地、可以直接开工的数据
- 需要你手动访问网页申请或查阅的数据与知识来源

## 1. 已下载到仓库的主线数据

当前主线优先使用 NHANES 2003-2004 周期的公开结构化数据。以下文件已经下载到：

`data/raw/nhanes/`

| 文件名 | 说明 | 行数 | 列数 | 本项目用途 |
| --- | --- | --- | --- | --- |
| `DEMO_C.XPT` | Demographic Variables and Sample Weights | 10122 | 44 | 年龄、性别、受试者基础信息 |
| `DIQ_C.XPT` | Diabetes Questionnaire | 9645 | 17 | 糖尿病相关病史与问卷变量 |
| `L10_C.XPT` | Glycohemoglobin | 6990 | 2 | HbA1c |
| `LEXAB_C.XPT` | Lower Extremity Disease - Ankle Brachial Blood Pressure Index | 3086 | 24 | ABI、下肢血供相关指标 |
| `LEXPN_C.XPT` | Lower Extremity Disease - Peripheral Neuropathy | 3086 | 46 | 周围神经病变相关检查变量 |
| `SMQ_C.XPT` | Smoking - Cigarette Use | 5041 | 42 | 吸烟相关变量 |

这些文件都可以通过 `SEQN` 进行受试者级关联。

## 2. NHANES 官方网页

建议你保留下面这些官方页面，后面做字段解释和答辩引用时会用到。

- NHANES 2003-2004 Examination Data 总入口  
  https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2003-2004
- NHANES 2003-2004 Demographics 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DEMO_C.htm
- NHANES 2003-2004 Diabetes Questionnaire 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DIQ_C.htm
- NHANES 2003-2004 Glycohemoglobin 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/L10_C.htm
- NHANES 2003-2004 Peripheral Neuropathy 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/LEXPN_C.htm
- NHANES 2003-2004 Ankle Brachial Blood Pressure Index 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/LEXAB_C.htm
- NHANES 2003-2004 Smoking 文档页  
  https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/SMQ_C.htm

## 3. 扩展数据与知识来源

### 3.1 CDC 足部健康页面

这是做知识图谱、护理宣教和就医建议时最稳的官方来源之一。

- CDC Promoting Foot Health  
  https://www.cdc.gov/diabetes/hcp/clinical-guidance/diabetes-podiatrist-health.html

### 3.2 DFUC 图像扩展

当前只建议作为扩展模块，不建议放进第一阶段主线。

- DFUC challenge report（Medical Image Analysis, 2024）  
  https://www.sciencedirect.com/science/article/pii/S1361841524000781

说明：

- 适合做足溃疡图像分割或异常区域演示。
- 数据属于受限公开，需按许可协议申请。
- 建议在主线页面跑通之后再接入。

### 3.3 热成像 / STANDUP 路线

当前建议作为接口预留和汇报中的扩展说明，不建议一开始就作为主线。

- Thermogram feature study discussing STANDUP and dataset heterogeneity  
  https://www.mdpi.com/2227-9059/11/12/3209
- Plantar Thermogram Database for the Study of Diabetic Foot Complications  
  https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications

说明：

- 更适合后续热成像异常提示或研究型分类任务。
- 数据处理流程更特定，前期不建议优先投入。

## 4. 当前推荐的数据使用顺序

1. 先用 `DEMO_C + DIQ_C + L10_C + LEXPN_C + LEXAB_C + SMQ_C` 跑通主线风险提示原型。
2. 再用 CDC 页面和种子图谱完善知识图谱问答。
3. 有时间再接 DFUC 图像演示。
4. 最后再考虑热成像和传感器接口。

## 5. 你现在可以直接做什么

你现在已经可以直接做以下事情：

- 读取 `data/raw/nhanes/*.XPT`
- 按 `SEQN` 合并结构化表
- 选出 README 中定义的主线字段
- 生成第一版 `processed/nhanes_risk_base.csv`
- 将处理后的结果接到风险问卷页

