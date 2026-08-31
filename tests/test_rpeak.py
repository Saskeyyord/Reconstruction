import numpy as np

from cycloscg.data.rpeak import detect_rpeaks


def test_rpeak_detection_on_synthetic_qrs_trace() -> None:
    fs = 256
    duration = 20
    time = np.arange(fs * duration) / fs
    rng = np.random.default_rng(19)
    ecg = 0.015 * rng.normal(size=len(time)) + 0.02 * np.sin(2 * np.pi * 0.2 * time)
    expected = np.arange(0.7, duration - 0.3, 0.8)
    for center in expected:
        ecg += 1.2 * np.exp(-0.5 * ((time - center) / 0.018) ** 2)
        ecg -= 0.25 * np.exp(-0.5 * ((time - center - 0.035) / 0.025) ** 2)
    detection = detect_rpeaks(ecg, fs, strict=True)
    assert detection.qc_passed
    assert abs(detection.heart_rate_bpm - 75.0) < 3.0
    assert abs(len(detection.indices) - len(expected)) <= 1

