import numpy as np

from cycloscg.data.mixing import achieved_snr_db, mix_at_snr


def test_synthetic_mixing_hits_requested_snr() -> None:
    rng = np.random.default_rng(7)
    clean = rng.normal(size=4096)
    noise = rng.normal(size=4096)
    for target in (-15.0, -10.0, -5.0, 0.0, 5.0):
        noisy, scale = mix_at_snr(clean, noise, target, polarity=-1.0)
        assert scale > 0
        assert abs(achieved_snr_db(clean, noisy) - target) < 1e-5

