# Cyclostationarity-Guided SCG Reconstruction

> GPU 实验已完成（2026-08-31）：当前推荐模型为
> `results/gpu_pilot_v2/checkpoints/best.pt`，完整结果与科学边界见
> [`docs/gpu_run_report.md`](docs/gpu_run_report.md)。

## 当前 GPU 推荐命令

```powershell
# 继续训练时使用验证集最佳模型作为初始化；不继承已衰减的优化器状态
python scripts/train_cycloscg.py `
  --config configs/cycloscg_gpu_v2.yaml `
  --init-checkpoint results/gpu_pilot_v2/checkpoints/best.pt `
  --output-root results/gpu_next

# 固定 held-out synthetic benchmark
python scripts/evaluate_synthetic.py `
  --config configs/cycloscg_gpu_v2.yaml `
  --cycloscg-checkpoint results/gpu_pilot_v2/checkpoints/best.pt `
  --baseline-config configs/baseline_gpu.yaml `
  --baseline-checkpoint results/baseline_gpu_pilot/checkpoints/best.pt `
  --samples-per-snr 64 `
  --output-root results/gpu_evaluation_v2 `
  --device cuda

# 仅评估 participant-held-out 的真实步行记录
python scripts/evaluate_walking.py `
  --config configs/cycloscg_gpu_v2.yaml `
  --cycloscg-checkpoint results/gpu_pilot_v2/checkpoints/best.pt `
  --baseline-config configs/baseline_gpu.yaml `
  --baseline-checkpoint results/baseline_gpu_pilot/checkpoints/best.pt `
  --output-root results/gpu_walking_v2 `
  --device cuda
```

本项目研究运动干扰下 SCG（seismocardiography）重构。核心问题不是简单频带分离，也不是把心脏视为“周期信号”、把运动视为“非周期噪声”，而是检验：

> 心源机械活动在 cardiac-phase 参考系中具有相对稳定的跨周期统计结构；运动干扰即使在绝对时间中呈周期性，通常也较少长期稳定锁定于 cardiac phase。

项目不使用 ICW-ConceFT。`ResidualUNet1D` 只作为 conventional direct-waveform baseline；proposed `CycloSCGNet` 的输入是 ECG R-wave 对齐后的 `[batch, cycles, cardiac_phase]` 矩阵。

## 当前实现

- manifest 驱动、列名兼容的标准化 CSV 读取；原始数据库保持只读。
- clean SCG 与 noise proxy 两套 participant-level train/validation/test 隔离，并保存固定 seed 的 `configs/splits.json`。
- 连续随机 SNR、随机噪声 offset、可选极性和 clean-input identity 的 dynamic mixing。
- ECG band-pass、QRS energy envelope、自适应 R 峰检测、漏峰/双峰修复与显式 QC。
- R-to-R cardiac-phase normalization；noisy input 和 clean target 强制共享同一组 R peaks。
- conventional waveform `ResidualUNet1D` baseline。
- `CycloSCGNet`：共享 beat encoder、phase-position 上的 cross-cycle attention、可解释 beat reliability、learnable cardiac consensus 和共享 residual decoder。
- `SmoothL1 + cross-cycle coherence + phase covariance/Gram + singular spectrum + multi-resolution STFT`；实验性 phase-lagged cyclic statistics 默认关闭。
- checkpoint save/resume、early stopping、CPU/单 CUDA、deterministic seed、CSV/TensorBoard 日志。
- held-out synthetic benchmark、clean identity preservation、real walking structural validation、cumulative ablation 和 gait-heart harmonic overlap 分析。
- Python/matplotlib 科研图：SVG、PDF、600-dpi TIFF 和 PNG 预览，输出到 `results/figures`。

## 数据与实验边界

默认入口为 `H:/数据库`，程序会自动定位其下唯一的 `dataset_manifest.csv`。实际数据库解析结果记录在 `docs/phase1_dataset_audit.md`。

- resting SCG 是 clean/silver-standard，用于独立 noise-proxy 的监督合成污染。
- walking SCG 包含真实心源成分与运动干扰，不是纯噪声。
- position recordings 是 motion-noise proxy，可能仍有少量心脏机械成分。
- 禁止以 `walking SCG -> resting SCG` 做逐点监督训练。
- walking evaluation 中，rest 只作为个体 cardiac-phase 结构分布参考，不是同步真值。
- cycle coherence 越高不自动等于越好；应同时比较 clean-rest 分布、waveform preservation、phase covariance 和 singular spectrum。

## 安装

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

要求 Python 3.10+、PyTorch、NumPy、SciPy、pandas、matplotlib 和 PyYAML。当前代码已在 Python 3.12 / CPU 上完成 smoke 验证。

## Phase 1–3：检查、划分与 baseline

```powershell
python scripts/inspect_dataset.py --database-root "H:\数据库"
python scripts/create_splits.py --database-root "H:\数据库" --seed 20260830

# 随机可视化 QC；同时保存 peaks、RR、HR
python scripts/verify_rpeaks.py --database-root "H:\数据库" --n 6

# 全量 ECG QC，不批量生成图片；非零退出表示存在需人工复核记录
python scripts/verify_rpeaks.py --database-root "H:\数据库" --n 0 --no-figures

python scripts/train_baseline.py --config configs/baseline.yaml
```

快速验证完整 baseline 管线：

```powershell
python scripts/train_baseline.py --config configs/baseline.yaml --smoke
```

## Phase 4–6：CycloSCGNet

```powershell
python scripts/train_cycloscg.py --config configs/cycloscg.yaml
```

快速验证 shared encoder、attention、consensus、完整损失与 checkpoint：

```powershell
python scripts/train_cycloscg.py --config configs/cycloscg.yaml --smoke
```

## Phase 7：held-out synthetic benchmark

```powershell
python scripts/evaluate_synthetic.py `
  --config configs/cycloscg.yaml `
  --baseline-config configs/baseline.yaml `
  --baseline-checkpoint results/baseline/checkpoints/best.pt `
  --cycloscg-checkpoint results/cycloscg/checkpoints/best.pt
```

输出逐样本和 `mean/std/n` 汇总，按 `-15/-10/-5/0 dB` 分层计算 RMSE、NRMSE、MAE、Pearson `r`、SNR improvement、PRD，以及 cycle/covariance/singular-spectrum 指标。`identity_preservation.csv` 单独报告 clean-input distortion。

## Phase 8：真实 walking external validation

```powershell
python scripts/evaluate_walking.py `
  --config configs/cycloscg.yaml `
  --cycloscg-checkpoint results/cycloscg/checkpoints/best.pt `
  --baseline-checkpoint results/baseline/checkpoints/best.pt
```

默认只评估 held-out SCG test subjects。记录若未通过 ECG QC，会写入 `walking_qc_exclusions.json`，不会静默继续。

## Phase 9：ablation 与 gait-heart overlap

```powershell
# 生成 A2–A7/Full 的固定控制配置
python scripts/run_ablation.py

# 正式顺序训练所有 proposed variants（耗时）
python scripts/run_ablation.py --run

python scripts/analyze_frequency_overlap.py `
  --config configs/cycloscg.yaml `
  --walking-metrics results/metrics/walking_structural_metrics.csv
```

`configs/ablation_matrix.yaml` 定义：A0 raw、A1 waveform U-Net、A2 cycle-aligned、A3 +attention、A4 +consensus、A5 +cycle loss、A6 +phase covariance、A7 +singular spectrum、Full +spectral/identity。所有变体复用同一 split、seed、sample budget 和 benchmark mixtures。

## 目录

```text
src/cycloscg/
  data/             # manifest, mixing, R peaks, phase warping, datasets, splits
  models/           # baseline U-Net, attention, consensus, CycloSCGNet
  losses/           # waveform and cyclostationarity-guided objectives
  metrics/          # waveform/cycle/phase-statistics metrics
  training/         # baseline and proposed trainers
  evaluation/       # synthetic, walking, ablation, frequency-overlap
  visualization/    # publication figure functions
scripts/            # command-line entry points
configs/            # baseline, full model, split, ablation matrix
tests/              # shape, leakage, SNR, losses, backward and mini-batch tests
results/            # checkpoints, metrics, figures and logs (generated files ignored by git)
```

## 结果解释

`--smoke` 仅验证工程正确性，训练样本和 epoch 极少，生成的数值与图不能用于科学结论。正式论文分析至少应完成全部配置训练、held-out benchmark、walking QC、多个 seed/置信区间，并在图注中报告 split、`n`、误差条定义和排除规则。
