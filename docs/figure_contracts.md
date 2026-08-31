# Figure contracts

## R-wave alignment QC

- Core conclusion: detected ECG R peaks provide defensible cardiac-cycle boundaries, and unreliable alignments are explicitly flagged.
- Archetype: quantitative QC trace.
- Output: Python, 180 mm width, editable SVG plus PDF and 600-dpi TIFF.
- Evidence: centered ECG trace, detected R-peak markers, heart rate, median RR, and QC status.
- Reviewer risk: a plausible mean heart rate can hide missed/double detections; relative RR outliers are therefore part of QC.

## Reconstruction and mechanism figures

- Core conclusion: cardiac-phase/cross-cycle modeling restores the clean target's structured variability more faithfully than direct-waveform denoising, including when gait and cardiac spectral components overlap.
- Archetype: asymmetric mixed-modality figure with a representative hero case and subordinate quantitative panels.
- Output: Python, 180 mm width, editable SVG plus PDF and 600-dpi TIFF; white background and a configurable, restrained method palette.
- Evidence hierarchy:
  - Hero: severe synthetic case and clean/noisy/reconstructed cardiac-phase matrices.
  - Validation: held-out synthetic metrics across SNR with participant-safe clean/noise splits.
  - Mechanism: reliability weights, cycle-correlation matrices, singular spectra, and gait-heart harmonic proximity.
  - Controls: clean-input identity and cumulative ablation from cycle alignment through structured losses.
- Statistics: show participant/sample `n`, define error bars in captions/source data, and keep per-sample CSVs.
- Integrity: no sample/column exclusion for visual convenience; all exclusions require QC reason and before/after counts.
- Reviewer risks:
  - Resting SCG is not synchronized walking ground truth; walking panels report distributional/structural proximity only.
  - Higher cycle coherence alone is not treated as better; it is interpreted relative to clean-rest structure and waveform preservation.
  - Reliability weights are model diagnostics, not proof of artifact identity.
  - Motion can be periodic; the claim concerns weaker consistency in the cardiac-phase reference frame, not non-periodicity.

