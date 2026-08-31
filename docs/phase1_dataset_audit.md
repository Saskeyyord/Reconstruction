# Phase 1 dataset and repository audit

审计日期：2026-08-31。数据库只读，所有新增代码和输出均位于当前 repository。

## 初始 repository 状态

- repository 初始为空；没有已有 preprocessing、model、training、README 或 `AGENTS.md` 可复用。
- 因此项目采用 `src/cycloscg` package layout 新建，但没有修改数据库源文件。

## 实际数据库结构

程序从配置的数据库父目录自动定位到唯一 manifest。完整扫描得到 129 条标准化记录：

| Category | Participants | Records |
|---|---:|---:|
| CleanRestSCG | 35 | 35 |
| RealWalkingContaminatedSCG | 35 | 70（1 step/s 与 2 steps/s 各 35） |
| MotionNoiseProxy | 6 | 24（4 个位置/participant） |

此外存在对应 raw-original copies、总索引工作簿、manifest、participant mapping、data dictionary、checksum 和 2520 条 fixed-SNR controlled-mix recipes。

总索引工作簿包含 7 个 sheet：总览、参与者映射、全部记录、数据集汇总、数据字典、受控混合配方和校验和；关键字段与 CSV manifest 一致，训练代码以便于版本控制和自动检查的 CSV manifest 为准。

## 标准化 CSV 实际格式

- 129/129 文件均为 5120 rows、12 columns、256 Hz、20 s。
- 全部标准化文件表头一致；未发现长度不匹配或目标列非有限值。
- 实际列：
  - `sample_index`
  - `time_from_recording_start_s`
  - `time_within_segment_s`
  - `accel_x_m_s2`, `accel_y_m_s2`, `accel_z_m_s2`
  - `accel_x_centered_m_s2`, `accel_y_centered_m_s2`, `accel_z_centered_m_s2`
  - `scg_z_bandpass_8_32Hz_m_s2`
  - `ecg_LA_RA_mV`, `ecg_LA_RA_centered_mV`
- 当前首选 SCG 列为 `scg_z_bandpass_8_32Hz_m_s2`，fallback 为 centered/raw Z acceleration；首选 ECG 列为 centered LA-RA。
- 原始 Shimmer CSV 实际为 tab-separated export，含 `sep=\t` 行、单位行和更长记录；训练默认读取已验证的标准 20 s CSV。

机器可读报告：`results/logs/dataset_audit.json`。

## Participant split

固定 seed `20260830`：

- clean SCG：25 train / 5 validation / 5 test participants。
- noise proxy：4 train / 1 validation / 1 test participants。
- clean 和 noise 两套 ID 在 train/validation/test 内均无交集。
- manifest 保存在 `configs/splits.json`；同一 SCG participant 的 rest/walking 不跨 split，同一 noise participant 的不同位置不跨 split。

## ECG R-peak QC

- 所有 35 条 clean-rest 记录通过自动 QC，可用于严格 cardiac-phase training。
- 全部 105 条胸口 ECG 中，102 条自动通过；3 条 walking 记录因疑似漏检/异常 RR ratio 被明确标为 manual review。
- 这 3 条不影响 supervised clean-rest training，但在 all-subject walking validation 中会被记录并排除，除非人工复核后修订 peak annotations。
- QC 保存 R indices、RR intervals、heart rate、pass/fail 原因；随机可视化输出 ECG+R peaks，完整审计可无图运行。

## 验证状态

- 数据库完整扫描：PASS（129/129）。
- unit tests：PASS（9 tests）。
- real-data cardiac-phase batch：PASS，shape `[2, 12, 256]`，finite。
- baseline CPU smoke training：PASS，1 epoch，best/last checkpoint、CSV/TensorBoard 已生成。
- CycloSCGNet full-loss CPU smoke training：PASS，1 epoch，best/last checkpoint、分项 loss 日志已生成。
- held-out synthetic/walking/frequency-overlap smoke pipelines：PASS。

Smoke 输出只证明代码可运行，不是论文性能结果。
