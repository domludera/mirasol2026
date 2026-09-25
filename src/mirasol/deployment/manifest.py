"""Provenance record for every exported ONNX graph.

``export_manifest.json`` ships alongside the ONNX Release assets and answers the
question the repository could not answer before: *which checkpoint is this
graph?*  Nothing recorded the source checkpoint, its fold or the exporter
versions, so the fold behind each published graph had to be inferred by
comparing patient sets after the fact.  One record per graph fixes that.

Each record carries the graph (path, size, sha256), its source checkpoint (path,
sha256, parsed epoch and monitored accuracy, experiment name, held-out fold),
the toolchain (opset, torch/onnx/onnxruntime/numpy versions), the calibration
draw for statically quantized graphs, and the measured hold-out prediction
spread that the constant-classifier guard produced.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from mirasol.deployment.calibration import CalibrationSet
from mirasol.paths import REPO_ROOT

MANIFEST_FILENAME: Final[str] = "export_manifest.json"

MANIFEST_NOTE: Final[str] = (
    "One record per exported ONNX graph. 'source_checkpoint_monitored_accuracy' "
    "is the value Lightning wrote into the checkpoint filename, i.e. accuracy on "
    "the held-out fold that was also monitored during training - see "
    "docs/errata.md on checkpoint selection optimism (Table S8). "
    "'holdout_prediction_counts' is [Present, Unknown, Absent] over the "
    "class-stratified hold-out segments; a single non-zero entry means the graph "
    "is a constant classifier and must not be deployed."
)

# Lightning writes checkpoints as e.g. 'l-epoch=142-acc=0.855.ckpt'.
_ACCURACY_RE: Final[re.Pattern[str]] = re.compile(r"acc=([0-9]*\.?[0-9]+)")
_EPOCH_RE: Final[re.Pattern[str]] = re.compile(r"epoch=([0-9]+)")
_FOLD_RE: Final[re.Pattern[str]] = re.compile(r"_fold([0-9]+)")


def sha256_of_file(path: Path) -> str:
    """sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Path string safe to publish in ``export_manifest.json``.

    The manifest ships as a GitHub Release asset, so it must not carry anyone's
    home directory.  Paths inside the repository become repo-relative; anything
    else under ``$HOME`` is abbreviated to ``~/...``.
    """
    resolved = path.resolve()
    for base, prefix in ((REPO_ROOT.resolve(), ""), (Path.home().resolve(), "~/")):
        try:
            return prefix + resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _run_directory(checkpoint_path: Path) -> str | None:
    """Experiment name implied by ``<exp>/checkpoint/lightning_logs/...`` paths."""
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoint":
            return parent.parent.name
    return None


@dataclass(frozen=True)
class CheckpointProvenance:
    """What a checkpoint file and its path say about the run that produced it."""

    path: str
    sha256: str
    epoch: int | None
    monitored_accuracy: float | None
    exp_name: str | None
    fold: int | None

    @classmethod
    def from_path(
        cls, checkpoint_path: Path, exp_name: str | None = None,
        fold: int | None = None,
    ) -> CheckpointProvenance:
        """Parse ``l-epoch=142-acc=0.855.ckpt`` and the run directory name.

        Args:
            checkpoint_path: the ``.ckpt`` being exported.
            exp_name: override the experiment name instead of inferring it.
            fold: override the held-out fold instead of inferring it from a
                ``_fold<N>`` suffix on the experiment name.

        Fields that cannot be determined stay ``None`` rather than being
        guessed, so a manifest never claims a fold it does not know.
        """
        name = checkpoint_path.name
        accuracy = _ACCURACY_RE.search(name)
        epoch = _EPOCH_RE.search(name)
        resolved_exp = exp_name or _run_directory(checkpoint_path)
        resolved_fold = fold
        if resolved_fold is None and resolved_exp is not None:
            match = _FOLD_RE.search(resolved_exp)
            resolved_fold = int(match.group(1)) if match else None
        return cls(
            path=portable_path(checkpoint_path),
            sha256=sha256_of_file(checkpoint_path),
            epoch=int(epoch.group(1)) if epoch else None,
            monitored_accuracy=float(accuracy.group(1)) if accuracy else None,
            exp_name=resolved_exp,
            fold=resolved_fold,
        )


def manifest_entry(
    model_type: str,
    variant: str,
    graph_path: Path,
    provenance: CheckpointProvenance,
    opset: int,
    calibration: CalibrationSet | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """One manifest record: graph, source checkpoint and toolchain."""
    import onnx
    import onnxruntime as ort
    import torch

    entry: dict[str, object] = {
        "architecture": model_type,
        "variant": variant,
        "path": portable_path(graph_path),
        "size_bytes": graph_path.stat().st_size,
        "sha256": sha256_of_file(graph_path),
        "source_checkpoint": provenance.path,
        "source_checkpoint_sha256": provenance.sha256,
        "source_checkpoint_epoch": provenance.epoch,
        "source_checkpoint_monitored_accuracy": provenance.monitored_accuracy,
        "exp_name": provenance.exp_name,
        "held_out_fold": provenance.fold,
        "opset": opset,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if calibration is not None:
        entry["calibration"] = {
            "dataset": portable_path(calibration.dataset_path),
            "n_segments": len(calibration.calibration),
            "seed": calibration.seed,
            "eval_fold_excluded": calibration.eval_fold,
        }
    if extra:
        entry.update(extra)
    return entry


def write_manifest(output_dir: Path, entries: Sequence[Mapping[str, object]]) -> Path:
    """Write ``export_manifest.json``, replacing records for the same variants.

    Re-exporting one variant must not drop the records of the others, and must
    not leave two records describing the same file.

    Raises:
        ValueError: an entry has no ``variant`` key, which would make merging
            ambiguous.
    """
    if any("variant" not in entry for entry in entries):
        raise ValueError("every manifest entry needs a 'variant' key")
    replaced = {entry["variant"] for entry in entries}
    manifest_path = output_dir / MANIFEST_FILENAME
    existing: list[Mapping[str, object]] = []
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = [
            record for record in loaded.get("graphs", [])
            if record.get("variant") not in replaced
        ]
    payload = {"note": MANIFEST_NOTE, "graphs": [*existing, *entries]}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
