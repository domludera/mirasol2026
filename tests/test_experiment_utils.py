import os

import pytest

from scripts import experiment_utils as eu

SAMPLE_CONFIG = '''import os

exp_name = "old_exp" # the saved ckpt prefix name of the model
model_type = "htsat"
# resume_checkpoint = "commented/out.ckpt"
resume_checkpoint = "some/path.ckpt"
esc_fold = 4 # fold comment
loss_weight = [4.0, 1.0]
sample_rate = 4000
clip_seconds = 30
clip_samples = sample_rate * clip_seconds
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point experiment_utils at a throwaway repo layout."""
    cfg = tmp_path / "config" / "esc_config.py"
    cfg.parent.mkdir()
    cfg.write_text(SAMPLE_CONFIG)
    monkeypatch.setattr(eu, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(eu, "CONFIG_FILE", str(cfg))
    return tmp_path


def _config_ns(repo):
    ns = {}
    exec((repo / "config" / "esc_config.py").read_text(), ns)
    return ns


def test_patch_config_rewrites_values_and_keeps_comments(repo):
    eu.patch_config(exp_name="new_exp", esc_fold=0, loss_weight=[4.0, 10.0, 1.0],
                    resume_checkpoint=None)
    text = (repo / "config" / "esc_config.py").read_text()
    ns = _config_ns(repo)

    assert ns["exp_name"] == "new_exp"
    assert ns["esc_fold"] == 0
    assert ns["loss_weight"] == [4.0, 10.0, 1.0]
    assert ns["resume_checkpoint"] is None
    assert 'exp_name = "new_exp" # the saved ckpt prefix name of the model' in text
    assert "esc_fold = 0 # fold comment" in text
    # commented-out assignments are left alone
    assert '# resume_checkpoint = "commented/out.ckpt"' in text


def test_patch_config_expr_is_written_verbatim(repo):
    eu.patch_config(clip_seconds=10, clip_samples=eu.Expr("sample_rate * clip_seconds"))
    assert _config_ns(repo)["clip_samples"] == 40000


def test_patch_config_unknown_key_raises(repo):
    with pytest.raises(KeyError):
        eu.patch_config(not_a_setting=1)


def _make_ckpts(repo, exp, version, names):
    d = repo / eu.RESULTS_DIR / exp / "checkpoint" / "lightning_logs" / f"version_{version}" / "checkpoints"
    d.mkdir(parents=True)
    paths = []
    for i, name in enumerate(names):
        p = d / name
        p.write_text("")
        os.utime(p, (1000 + i, 1000 + i))
        paths.append(p)
    return paths


def test_find_best_checkpoint_missing_experiment(repo):
    assert eu.find_best_checkpoint("nope") is None


def test_find_best_checkpoint_by_acc_uses_latest_version_numerically(repo):
    _make_ckpts(repo, "exp", 2, ["l-epoch=1-acc=0.990.ckpt"])
    _make_ckpts(repo, "exp", 10, ["l-epoch=5-acc=0.800.ckpt", "l-epoch=9-acc=0.850.ckpt"])
    best = eu.find_best_checkpoint("exp")
    assert best == os.path.join(eu.RESULTS_DIR, "exp", "checkpoint", "lightning_logs",
                                "version_10", "checkpoints", "l-epoch=9-acc=0.850.ckpt")


def test_find_best_checkpoint_by_mtime(repo):
    _make_ckpts(repo, "exp", 0, ["l-epoch=9-acc=0.900.ckpt", "last.ckpt"])
    assert eu.find_best_checkpoint("exp", by="mtime").endswith("last.ckpt")


def test_predictions_exist(repo):
    assert not eu.predictions_exist("exp")
    preds = repo / eu.PREDS_DIR
    preds.mkdir(parents=True)
    (preds / "exp.data.csv").write_text("")
    assert eu.predictions_exist("exp")


def test_train_and_test_happy_path(repo, monkeypatch):
    calls = []

    def fake_run(cmd, desc=""):
        calls.append(cmd[-1])
        if cmd[-1] == "train":
            _make_ckpts(repo, "exp", 0, ["l-epoch=3-acc=0.700.ckpt"])
        return 0

    monkeypatch.setattr(eu, "run_cmd", fake_run)
    status = eu.train_and_test("exp", model_type="tinyvit", esc_fold=1)

    assert status == "ok"
    assert calls == ["train", "test"]
    ns = _config_ns(repo)
    assert ns["exp_name"] == "exp"
    assert ns["model_type"] == "tinyvit"
    assert ns["resume_checkpoint"].endswith("l-epoch=3-acc=0.700.ckpt")


def test_train_and_test_stops_on_train_failure(repo, monkeypatch):
    monkeypatch.setattr(eu, "run_cmd", lambda cmd, desc="": 1)
    assert eu.train_and_test("exp") == "train_failed"
    assert _config_ns(repo)["resume_checkpoint"] is None


def test_train_and_test_no_checkpoint(repo, monkeypatch):
    monkeypatch.setattr(eu, "run_cmd", lambda cmd, desc="": 0)
    assert eu.train_and_test("exp") == "no_checkpoint"


def test_real_config_has_every_task_key():
    """All keys runners patch must exist in the real config file."""
    with open(os.path.join(os.path.dirname(eu.CONFIG_FILE), "esc_config.py")) as f:
        text = f.read()
    keys = {"exp_name", "model_type", "esc_fold", "resume_checkpoint",
            "clip_seconds", "clip_samples"}
    for task in eu.TASKS.values():
        keys |= set(task)
    for key in keys:
        assert f"\n{key} =" in text, key
