import pytest

from src.prepare_data import parse_metadata, patient_level_folds

CSV = """Patient ID,Recording locations:,Murmur
1,AV+MV,Present
2,PV,Unknown
3,TV+AV,Absent
"""


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "training_data.csv"
    p.write_text(CSV)
    return str(p)


def test_3class_labels_are_present_unknown_absent(csv_file):
    meta = parse_metadata(csv_file, "3class")
    assert {k: v["murmur"] for k, v in meta.items()} == {
        "1_AV.wav": 0, "1_MV.wav": 0, "2_PV.wav": 1, "3_TV.wav": 2, "3_AV.wav": 2}


def test_default_task_is_3class(csv_file):
    assert parse_metadata(csv_file) == parse_metadata(csv_file, "3class")


def test_binary_merges_unknown_into_present(csv_file):
    meta = parse_metadata(csv_file, "binary")
    assert meta["2_PV.wav"]["murmur"] == 0
    assert meta["3_TV.wav"]["murmur"] == 1


def test_folds_keep_patients_together():
    samples = [{"name": f"{p}_{loc}.wav", "patient": str(p), "target": 0}
               for p in range(20) for loc in ("AV", "MV")]
    folds = patient_level_folds(samples, n_folds=5, seed=42)
    fold_of = {}
    for k, fold in enumerate(folds):
        for s in fold:
            assert fold_of.setdefault(s["patient"], k) == k
    assert sum(len(f) for f in folds) == len(samples)


def test_default_output_names_do_not_collide():
    from src.prepare_data import default_output_name
    assert default_output_name("3class", 30) == "train-circor-data-30s.npy"
    assert default_output_name("binary", 30) == "train-circor-data-30s-binary.npy"
