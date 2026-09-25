"""Shared fixtures and reference implementations for the MIRASOL test suite.

The suite defends the artifacts the paper promises: the five-fold patient
partition (``data/splits/folds.csv``), the out-of-fold prediction dumps in
``results/predictions/`` from which Table 1, Table 2 and Tables S1-S8 are
regenerated, and the single definition of the network's input representation
(``mirasol.signal.mel``) together with its vendored edge copy.

Constraints enforced here rather than in each module:

* Nothing requires the CirCor dataset, a GPU, PyTorch or an ONNX file.  The
  two tests that need CirCor's ``training_data.csv`` skip cleanly without it.
* Expected values are stated independently of the code under test - either
  transcribed from the camera-ready or recomputed here by a second
  implementation (:func:`reference_fold_assignment`,
  :func:`reference_patient_pool`).
* No absolute path is hardcoded.  Locations come from :mod:`mirasol.paths`
  (repo root from ``__file__``, overridable by environment variable), and the
  pre-release working tree used for the legacy parity checks is located
  through ``MIRASOL_LEGACY_SOURCE`` only.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import random
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SRC_DIR: Final[Path] = REPO_ROOT / "src"

# A fresh clone must be testable before ``pip install -e .``, and the analysis
# scripts are imported as ``analysis.<module>`` from the repository root.
for _entry in (REPO_ROOT, SRC_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from mirasol import paths as mirasol_paths  # noqa: E402 - needs the sys.path above

PATHS: Final[mirasol_paths.Paths] = mirasol_paths.default_paths()

EDGE_DIR: Final[Path] = REPO_ROOT / "deployment" / "edge"
EDGE_MEL: Final[Path] = EDGE_DIR / "mel.py"
BENCH_INPUT: Final[Path] = EDGE_DIR / "bench_input.npy"

LEGACY_SOURCE_ENV: Final[str] = "MIRASOL_LEGACY_SOURCE"

# --------------------------------------------------------------------------
# Measured properties of the released partition (docs/splits.md)
# --------------------------------------------------------------------------
N_FOLDS: Final[int] = 5
SPLIT_SEED: Final[int] = 42
N_PATIENTS: Final[int] = 941
FOLD_PATIENT_COUNTS: Final[tuple[int, ...]] = (189, 188, 188, 188, 188)
FOLD_SEGMENT_COUNTS: Final[tuple[int, ...]] = (739, 747, 756, 753, 726)

#: Reading the supplement literally (sorted identifiers, no shuffle) puts this
#: many of the 941 patients in the wrong fold.  Measured, not quoted.
LITERAL_READING_MISMATCHES: Final[int] = 742

#: Dropped during dataset preparation: all four of its recordings use the
#: duplicate-location naming 50321_AV_1/_2, 50321_MV_1/_2.
EXCLUDED_PATIENT: Final[str] = "50321"

#: Table 1 / Table 2 column order: ascending parameter count.
THREE_CLASS_MODELS: Final[tuple[tuple[str, str], ...]] = (
    ("MobileNetV3-S", "mobilenetv3_3class"),
    ("FastViT", "fastvit_3class"),
    ("EfficientNet-B0", "efficientnet_b0_3class"),
    ("TinyViT", "tinyvit_3class"),
    ("ResNet-18", "resnet18_3class"),
    ("HTS-AT", "htsat_3class"),
)

#: Table S7 ran four of the six architectures in the two-class configuration.
BINARY_MODELS: Final[tuple[str, ...]] = (
    "efficientnet_b0_binary",
    "fastvit_binary",
    "tinyvit_binary",
    "htsat_binary",
)

PRESENT: Final[int] = 0
UNKNOWN: Final[int] = 1
ABSENT: Final[int] = 2


# --------------------------------------------------------------------------
# Artifact locations
# --------------------------------------------------------------------------
def predictions_dir() -> Path:
    """Directory holding the committed out-of-fold prediction dumps."""
    return PATHS.preds_dir


def prediction_csv(prefix: str, fold: int | None = None) -> Path:
    """Path of one prediction dump, e.g. ``htsat_3class_fold2.data.csv``."""
    stem = prefix if fold is None else f"{prefix}_fold{fold}"
    return predictions_dir() / f"{stem}.data.csv"


@pytest.fixture(scope="session")
def preds_dir() -> Path:
    """Committed prediction dumps; a missing directory is a release defect."""
    directory = predictions_dir()
    if not directory.is_dir():
        pytest.fail(
            f"{directory.name}/ is missing: results/predictions/ is committed "
            "evidence that every table script depends on"
        )
    return directory


def require_circor_meta() -> Path:
    """Return CirCor's ``training_data.csv`` or skip the calling test."""
    meta = PATHS.circor_meta
    if not meta.is_file():
        pytest.skip(
            "CirCor training_data.csv is not present (it is never "
            f"redistributed); download it from {mirasol_paths.PHYSIONET_URL} "
            "and set CIRCOR_ROOT to run this test"
        )
    return meta


# --------------------------------------------------------------------------
# Reference implementations, deliberately independent of the package
# --------------------------------------------------------------------------
def reference_fold_assignment(
    patient_ids: Iterable[str],
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    """Sort (as strings) -> ``random.Random(seed).shuffle`` -> round-robin.

    Transcribed from ``src/prepare_data.py`` of the pre-release tree.  The
    supplement's prose omits the shuffle, which is why the folds ship as a CSV;
    this is the executable form of the recipe in ``docs/splits.md``.
    """
    patients = sorted(str(pid) for pid in patient_ids)
    random.Random(seed).shuffle(patients)
    return {pid: index % n_folds for index, pid in enumerate(patients)}


def literal_reading_fold_assignment(
    patient_ids: Iterable[str], n_folds: int = N_FOLDS
) -> dict[str, int]:
    """The supplement read literally: sorted identifiers, no shuffle."""
    patients = sorted(str(pid) for pid in patient_ids)
    return {pid: index % n_folds for index, pid in enumerate(patients)}


def read_prediction_frame(path: Path) -> pd.DataFrame:
    """Read a headerless prediction dump into a frame with a patient column."""
    frame = pd.read_csv(path, header=None)
    n_classes = frame.shape[1] - 2
    frame.columns = [f"logit{i}" for i in range(n_classes)] + ["target", "filename"]
    frame["filename"] = frame["filename"].astype(str)
    frame["patient"] = frame["filename"].str.split("_").str[0]
    return frame


def reference_patient_pool(path: Path) -> pd.DataFrame:
    """Mean-pool segment logits per patient, then argmax - the paper's rule.

    A second implementation of ``mirasol.metrics.pool_by_patient`` written with
    pandas, so the golden tables have an opinion that does not come from the
    module they are meant to police.  Not interchangeable with mean-pooled
    softmax or segment majority vote: HTS-AT's published Challenge Score of
    .875 becomes .869 and .846 under those two rules.
    """
    frame = read_prediction_frame(path)
    logit_columns = [c for c in frame.columns if c.startswith("logit")]
    pooled = (
        frame.groupby("patient", as_index=False)
        .agg({**{c: "mean" for c in logit_columns}, "target": "first"})
        .sort_values("patient", kind="mergesort")
        .reset_index(drop=True)
    )
    values = pooled[logit_columns].to_numpy(dtype=float)
    return pooled.assign(
        pred=np.argmax(values, axis=1), target=pooled["target"].astype(int)
    )


# --------------------------------------------------------------------------
# Deterministic synthetic signals
# --------------------------------------------------------------------------
def bench_waveform(clip_samples: int = 120_000, clip_seconds: int = 30) -> np.ndarray:
    """The exact synthetic waveform behind ``deployment/edge/bench_input.npy``.

    Reproduces ``generate_bench_input.py`` of the pre-release tree, whose output
    is the only input to the Table 4 on-device benchmark: white noise at 0.01
    plus 50 Hz and 120 Hz tones, all in float32, seeded with
    ``np.random.RandomState(42)``.  The float32 arithmetic and the ``linspace``
    time base are load-bearing for bit-exact reproduction.
    """
    waveform = np.random.RandomState(42).randn(clip_samples).astype(np.float32) * 0.01
    time = np.linspace(0, clip_seconds, clip_samples, dtype=np.float32)
    waveform += 0.1 * np.sin(2 * np.pi * 50 * time)
    waveform += 0.05 * np.sin(2 * np.pi * 120 * time)
    return waveform


def synthetic_waveform(n_samples: int, sample_rate: int = 4000) -> np.ndarray:
    """Fixed synthetic phonocardiogram-like waveform, float32.

    A 1.2 Hz envelope over two heart-sound bands, a component above the
    bandpass, and a small deterministic noise floor, so the comparisons
    exercise both filled and near-empty mel bins.
    """
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 1.2 * time))
    signal = envelope * (
        0.6 * np.sin(2.0 * np.pi * 45.0 * time)
        + 0.3 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.1 * np.sin(2.0 * np.pi * 620.0 * time)
    )
    noise = np.random.default_rng(20260826).standard_normal(n_samples) * 0.01
    return (signal + noise).astype(np.float32)


# --------------------------------------------------------------------------
# Loading modules that are not importable as part of the package
# --------------------------------------------------------------------------
def load_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Import a module by file path (``deployment/edge/`` is not a package)."""
    if not path.is_file():
        pytest.fail(f"{path.name} is missing from deployment/edge/")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def edge_mel() -> types.ModuleType:
    """The vendored, torch-free front end shipped to the Orange Pi."""
    return load_module_from_path("mirasol_edge_mel_under_test", EDGE_MEL)


# --------------------------------------------------------------------------
# The pre-release working tree (parity checks only)
# --------------------------------------------------------------------------
def legacy_source_root() -> Path | None:
    """Root of the original private working tree, or ``None``."""
    override = os.environ.get(LEGACY_SOURCE_ENV, "").strip()
    if not override:
        return None
    root = Path(override)
    return root if root.is_dir() else None


def require_legacy_source() -> Path:
    """Return the legacy source root or skip the calling test.

    The pre-release tree is not part of the release, so a cloner's CI run skips
    these checks; the parity anchor that always runs is the committed
    ``bench_input.npy``.
    """
    root = legacy_source_root()
    if root is None:
        pytest.skip(
            f"set {LEGACY_SOURCE_ENV} to the pre-release working tree to run the "
            "legacy parity checks (not required for a clone)"
        )
    return root


def legacy_config(root: Path) -> types.SimpleNamespace:
    """Literal assignments of the pre-release ``config/esc_config.py``.

    Parsed rather than imported, so the parity tests never pull in torch and so
    the compared values are exactly those the published exports used.
    """
    path = root / "config" / "esc_config.py"
    if not path.is_file():
        pytest.skip(f"{path.name} is unavailable in the legacy tree")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue  # os.path.join(...) and friends: not needed here
    return types.SimpleNamespace(**values)


def load_legacy_function(
    root: Path,
    relative_path: str,
    function_name: str,
    globals_: Mapping[str, Any],
) -> Callable[..., Any]:
    """Compile one function out of a legacy module without importing it.

    The pre-release modules import torch, timm and onnxruntime at module scope;
    extracting a single function definition through :mod:`ast` keeps the parity
    tests torch-free while pinning the exact code the published numbers came
    from.
    """
    path = root / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} is unavailable in the legacy tree")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, Any] = dict(globals_)
            exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
            return namespace[function_name]
    pytest.skip(f"{relative_path} no longer defines {function_name}()")
