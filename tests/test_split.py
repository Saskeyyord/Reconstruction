from cycloscg.data.split import ParticipantSplit, create_participant_split


def test_participant_split_has_no_leakage_and_is_reproducible(tmp_path) -> None:
    clean = [f"SCG_P{i:03d}" for i in range(1, 36)]
    noise = [f"NOISE_P{i:03d}" for i in range(1, 7)]
    first = create_participant_split(clean, noise, seed=123)
    second = create_participant_split(clean, noise, seed=123)
    assert first == second
    first.validate()
    assert (len(first.train_clean_subjects), len(first.val_clean_subjects), len(first.test_clean_subjects)) == (
        25,
        5,
        5,
    )
    assert (len(first.train_noise_subjects), len(first.val_noise_subjects), len(first.test_noise_subjects)) == (
        4,
        1,
        1,
    )
    path = tmp_path / "splits.json"
    first.save(path)
    assert ParticipantSplit.load(path) == first

