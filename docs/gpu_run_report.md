# GPU training and evaluation report

Run date: 2026-08-31  
Hardware: NVIDIA GeForce RTX 3070, 8 GB  
Runtime: PyTorch 2.13.0+cu130, automatic mixed precision, deterministic seed 20260830

## Model selection

The recommended checkpoint is:

```text
results/gpu_pilot_v2/checkpoints/best.pt
```

It was initialized from the best v1 checkpoint and fine-tuned with a fresh
optimizer on a low-SNR-focused dynamic distribution. Selection used only the
synthetic validation split. Real-walking test subjects were not used for model
selection or hyperparameter tuning.

## Held-out synthetic benchmark

Each cell is the mean of 64 fixed participant-held-out mixtures. Lower RMSE and
cycle error are better; higher Pearson correlation and SNR improvement are
better.

| Input SNR | Method | RMSE | Pearson r | SNR improvement (dB) | Cycle error |
|---:|---|---:|---:|---:|---:|
| -15 dB | Waveform U-Net | 0.9886 | 0.1548 | 15.0145 | 0.6490 |
| -15 dB | CycloSCGNet v2 | **0.9069** | **0.3870** | **15.7999** | **0.3489** |
| -10 dB | Waveform U-Net | 0.9436 | 0.3005 | 10.4185 | 0.5999 |
| -10 dB | CycloSCGNet v2 | **0.8006** | **0.5725** | **11.9156** | **0.3042** |
| -5 dB | Waveform U-Net | 0.8451 | 0.5206 | 6.3971 | 0.4848 |
| -5 dB | CycloSCGNet v2 | **0.6298** | **0.7621** | **9.0544** | **0.2055** |
| 0 dB | Waveform U-Net | 0.6476 | 0.7586 | 3.7159 | 0.2789 |
| 0 dB | CycloSCGNet v2 | **0.5065** | **0.8552** | **5.9154** | **0.1592** |

CycloSCGNet v2 improves RMSE, correlation, SNR improvement, cycle-coherence
error, and singular-spectrum distance over the waveform baseline at every SNR.
For phase-Gram distance, it is better at -10/-5/0 dB but worse at -15 dB
(1.1305 versus 1.0361). This extreme-SNR exception is retained rather than
hidden.

Clean-input preservation for v2 is RMSE 0.00745, Pearson r 0.99997, PRD 0.745%,
and output SNR 43.20 dB.

Detailed tables:

- `results/gpu_evaluation_v2/metrics/synthetic_per_sample.csv`
- `results/gpu_evaluation_v2/metrics/synthetic_summary.csv`
- `results/gpu_evaluation_v2/metrics/identity_preservation.csv`

## Real-walking external validation

The evaluation contains five held-out SCG participants, two walking conditions
per participant, and zero ECG-QC exclusions. Rest is an individual structural
reference distribution, not synchronized walking ground truth.

Compared with raw walking, CycloSCGNet moves the mean pairwise/template
statistics toward the participant's rest distribution in both conditions:

| Condition | Method | Pairwise distance to rest | Template distance to rest | Variability relative error |
|---|---|---:|---:|---:|
| 1 step/s | Raw walking | 0.2078 | 0.1664 | 0.3367 |
| 1 step/s | Waveform U-Net | 0.2041 | **0.1600** | 0.3792 |
| 1 step/s | CycloSCGNet v2 | **0.2040** | 0.1613 | **0.3053** |
| 2 steps/s | Raw walking | 0.1982 | 0.1962 | 0.3615 |
| 2 steps/s | Waveform U-Net | **0.1870** | 0.1897 | 0.3404 |
| 2 steps/s | CycloSCGNet v2 | 0.1938 | **0.1878** | **0.3037** |

The proposed model is consistently better than raw walking but does not dominate
the waveform baseline on every surrogate structural metric. Because no
synchronized clean walking target exists, these results support structural
plausibility rather than pointwise reconstruction accuracy.

The gait-heart overlap analysis included 10 records. The exploratory Spearman
association between harmonic gap and CycloSCGNet advantage was rho=-0.049,
p=0.894. This small held-out sample does not support a frequency-overlap trend.

Detailed tables:

- `results/gpu_walking_v2/metrics/walking_structural_metrics.csv`
- `results/gpu_walking_v2/metrics/gait_heart_overlap.csv`
- `results/gpu_walking_v2/metrics/gait_heart_overlap_statistics.json`

## Controlled GPU ablation

All trainable variants used the same participant split, seed, architecture
width, 30 epochs, 512 dynamic training samples per epoch, and fixed benchmark
recipes. The table averages four SNR levels with 32 samples per level.

| Variant | RMSE | Pearson r | SNR improvement | Cycle error | Phase-Gram distance | Identity PRD |
|---|---:|---:|---:|---:|---:|---:|
| A0 Raw | 2.8615 | 0.4220 | 0.0000 | 0.5475 | 0.8475 | n/a |
| A1 Waveform U-Net | 0.8591 | 0.4330 | 8.8465 | 0.5088 | 0.8382 | n/a |
| A2 Cycle aligned | 0.8497 | 0.4719 | 8.9500 | 0.4026 | 0.8726 | 16.58% |
| A3 + Attention | 0.7279 | 0.6280 | 10.4631 | 0.2856 | 0.8738 | 17.19% |
| A4 + Consensus | **0.7074** | **0.6503** | **10.7271** | 0.2976 | 0.8831 | 8.18% |
| A5 + Cycle loss | 0.7146 | 0.6460 | 10.6128 | **0.2590** | 0.8332 | 7.79% |
| A6 + Covariance loss | 0.7185 | 0.6394 | 10.5787 | 0.2596 | **0.8150** | 7.87% |
| A7 + Singular loss | 0.7190 | 0.6390 | 10.5749 | 0.2600 | 0.8165 | 7.78% |
| Full + Spectral/identity | 0.7385 | 0.6226 | 10.3424 | 0.2737 | 0.8536 | **1.73%** |

The cumulative experiment shows that cross-cycle attention provides the largest
reconstruction gain, consensus adds a further gain, and structured losses improve
their targeted statistics. The controlled Full model sacrifices some denoising
score to greatly reduce clean-waveform distortion. The subsequent v2 optimization
improves this trade-off further (0.745% identity PRD while outperforming the
waveform baseline at every benchmark SNR).

Detailed tables and figures:

- `results/gpu_ablation/metrics/ablation_summary.csv`
- `results/gpu_ablation/metrics/ablation_summary_by_snr.csv`
- `results/gpu_ablation/metrics/ablation_identity_summary.csv`
- `results/gpu_ablation/figures/`

## Verification

- CUDA forward, backward, AMP, SVD/Gram/STFT loss, checkpoint, and resume paths
  completed without OOM or non-finite values.
- Peak allocated GPU memory during controlled ablation was approximately 0.37 GB.
- Unit tests: 9 passed.
