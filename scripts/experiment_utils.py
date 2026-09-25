"""Shared helpers for experiment runner scripts.

The training entry point (main_moody.py) reads all settings from
config/esc_config.py, so runners work by patching that file, launching
main_moody.py in a subprocess, and locating the resulting checkpoints.

All paths are relative to the repository root, and subprocesses run there.
"""
import glob
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(REPO_ROOT, "config", "esc_config.py")
RESULTS_DIR = os.path.join("data", "results")
PREDS_DIR = os.path.join("outputs", "preds_targets")
PYTHON = sys.executable

# Dataset / label settings for each task formulation.
TASKS = {
    "binary": {
        "dataset_npy_name": "train-circor-data-30s-binary.npy",
        "classes_num": 2,
        "loss_weight": [4.0, 1.0],
    },
    "3class": {
        "dataset_npy_name": "train-circor-data-30s.npy",
        "classes_num": 3,
        "loss_weight": [4.0, 10.0, 1.0],
    },
}


class Expr(str):
    """A config value written verbatim as a Python expression (not quoted)."""


def _render(value) -> str:
    if isinstance(value, Expr):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def patch_config(**overrides) -> None:
    """Rewrite top-level assignments in esc_config.py, keeping trailing comments.

    Raises:
        KeyError: if a key has no top-level assignment in the config file.
    """
    with open(CONFIG_FILE, "r") as f:
        content = f.read()

    for key, value in overrides.items():
        line = f"{key} = {_render(value)}"
        pattern = rf"^{re.escape(key)}\s*=[^#\n]*?(\s*#.*)?$"
        content, n = re.subn(pattern, lambda m: line + (m.group(1) or ""),
                             content, flags=re.MULTILINE)
        if n == 0:
            raise KeyError(f"'{key}' not found in {CONFIG_FILE}")

    with open(CONFIG_FILE, "w") as f:
        f.write(content)

    summary = ", ".join(f"{k}={v}" for k, v in overrides.items())
    print(f"  Config patched: {summary}")


def _version_number(path: str) -> int:
    m = re.search(r"version_(\d+)$", path)
    return int(m.group(1)) if m else -1


def _checkpoint_acc(path: str) -> float:
    m = re.search(r"acc=(\d+\.\d+)", os.path.basename(path))
    return float(m.group(1)) if m else 0.0


def find_best_checkpoint(exp_name: str, by: str = "acc") -> str | None:
    """Return a checkpoint from the latest lightning_logs version of an experiment.

    Args:
        exp_name: Experiment name (directory under data/results/).
        by: "acc" picks the highest acc=... in the filename;
            "mtime" picks the most recently written file.
    """
    base = os.path.join(RESULTS_DIR, exp_name, "checkpoint", "lightning_logs")
    versions = glob.glob(os.path.join(REPO_ROOT, base, "version_*"))
    if not versions:
        return None

    latest = max(versions, key=_version_number)
    ckpts = glob.glob(os.path.join(latest, "checkpoints", "*.ckpt"))
    if not ckpts:
        return None

    key = os.path.getmtime if by == "mtime" else _checkpoint_acc
    return os.path.relpath(max(ckpts, key=key), REPO_ROOT)


def predictions_exist(exp_name: str) -> bool:
    """True if main_moody.py test has already written predictions for exp_name."""
    return os.path.exists(os.path.join(REPO_ROOT, PREDS_DIR, f"{exp_name}.data.csv"))


def run_cmd(cmd: list, desc: str = "") -> int:
    """Run a command from the repo root, streaming output. Returns the exit code."""
    print(f"\n  >>> {' '.join(cmd)}")
    if desc:
        print(f"  ({desc})")
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        print(f"  WARNING: Command exited with code {rc}")
    return rc


def train_and_test(exp_name: str, ckpt_by: str = "acc", **overrides) -> str:
    """Patch config, train, then test on the best checkpoint.

    Returns:
        "ok", "train_failed", "no_checkpoint" or "test_failed".
    """
    patch_config(exp_name=exp_name, resume_checkpoint=None, **overrides)

    if run_cmd([PYTHON, "main_moody.py", "train"], f"Training {exp_name}") != 0:
        print(f"  FAILED: training {exp_name}")
        return "train_failed"

    ckpt = find_best_checkpoint(exp_name, by=ckpt_by)
    if not ckpt:
        print(f"  WARNING: no checkpoint found for {exp_name}")
        return "no_checkpoint"
    print(f"  Best checkpoint: {ckpt}")
    patch_config(resume_checkpoint=ckpt)

    if run_cmd([PYTHON, "main_moody.py", "test"], f"Testing {exp_name}") != 0:
        print(f"  FAILED: testing {exp_name}")
        return "test_failed"
    return "ok"
