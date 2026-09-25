# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The Model Training Wrapper
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Lightning wrapper that trains and evaluates every architecture in the paper.

``SEDWrapper`` owns the optimiser, the learning-rate warm-up schedule and the
segment-level metrics for all six architectures of Table 1, so MobileNetV3-S and
HTS-AT are trained by identical code -- the "identical preprocessing,
augmentation and training conditions" claim of Sec. 2.2 is implemented here.

Paper artifacts served: the checkpoints behind Table 1, Table 2, Tables S1-S8
and Table S10, and (through :class:`mirasol.training.predictions.PredictionWriter`)
the segment-level dumps every one of those tables is aggregated from.

Metric scope: this module logs *segment-level* accuracy and macro F1 only.
Accuracy is what ``ModelCheckpoint`` monitors; the Challenge Score reported in
the paper is a *patient-level* quantity computed post hoc from the prediction
dumps by :mod:`mirasol.metrics`, never during training.

Removed from the upstream file, all of it unreachable on the CirCor path:
``Ensemble_SEDWrapper`` (only ever constructed by a broken ``esm_test``
subcommand), the four ``if "challenge_score" in metric_dict`` branches (nothing
ever put that key in the dict), every ``dataset_type == 'audioset'`` and
``'scv2'`` metric branch, the AudioSet-only 10-shift test-time augmentation, the
framewise-localisation branch (which read config keys that do not exist), and
``from torchcontrib.optim import SWA`` -- an unmaintained dependency imported for
two commented-out lines.
"""

from __future__ import annotations

import bisect
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from mirasol.training.losses import get_loss_func
from mirasol.training.predictions import AUDIO_NAME_KEY, LOGITS_KEY, TARGET_KEY
from mirasol.training.utils import get_mix_lambda

__all__ = ["SEDWrapper"]

#: Mixup alpha used when mixup is explicitly enabled; the upstream AudioSet
#: value. See the note in :meth:`SEDWrapper.training_step`.
DEFAULT_MIXUP_ALPHA = 0.5


class SEDWrapper(pl.LightningModule):
    """Train and evaluate one audio classifier on 30 s CirCor segments.

    Args:
        sed_model: a :class:`mirasol.models.base.BaseAudioModel` returning a dict
            with ``clipwise_output`` and ``framewise_output``.
        config: resolved configuration (see :mod:`mirasol.config.resolve`).
        dataset: the training dataset, whose sampling queue is regenerated after
            each epoch. ``None`` is allowed for evaluation-only use, such as the
            quantization-aware-training path.
    """

    def __init__(
        self, sed_model: torch.nn.Module, config: Any, dataset: Any = None
    ) -> None:
        super().__init__()
        if getattr(config, "fl_local", False):
            raise ValueError(
                "config.fl_local is not supported: framewise sound-event "
                "localisation is an upstream HTS-AT feature that needs heatmap_dir "
                "and test_file keys, and it is not part of the murmur-screening "
                "protocol. Set fl_local=False."
            )
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
        loss_weight = getattr(config, "loss_weight", None)
        if loss_weight is not None:
            # A buffer so the weights follow the module across devices.
            self.register_buffer("loss_weight", torch.FloatTensor(loss_weight))
        else:
            self.loss_weight = None
        self.loss_func = get_loss_func(config.loss_type)
        self.validation_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.test_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    # -- metrics ---------------------------------------------------------- #
    def evaluate_metric(self, pred: np.ndarray, ans: np.ndarray) -> dict[str, float]:
        """Segment-level accuracy and F1 from logits and integer targets.

        F1 is binary for the two-class configuration of Table S7 and macro for
        the three-class configuration used everywhere else.
        """
        preds_bin = np.argmax(pred, 1)
        acc = accuracy_score(ans, preds_bin)
        average = "binary" if self.config.classes_num == 2 else "macro"
        f1 = f1_score(ans, preds_bin, average=average, zero_division=0)
        return {"acc": float(acc), "f1": float(f1)}

    # -- forward ---------------------------------------------------------- #
    def forward(
        self, x: torch.Tensor, mix_lambda: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(clipwise_logits, framewise_output)`` for a waveform batch."""
        output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Run a single numpy waveform batch through the model in eval mode."""
        device = next(self.parameters()).device
        self.eval()
        tensor = torch.from_numpy(x).float().to(device)
        with torch.no_grad():
            output_dict = self.sed_model(tensor, None, True)
        return {key: value.detach().cpu().numpy() for key, value in output_dict.items()}

    # -- training --------------------------------------------------------- #
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """One class-weighted cross-entropy step on a batch of segments.

        Mixup note: upstream HTS-AT draws mixup coefficients only when
        ``dataset_type == 'audioset'``, so on the CirCor path ``mix_lambda`` was
        always ``None`` and the models behind the published tables were trained
        with SpecAugment but *without* Mixup, despite Sec. 2.2 listing it. That
        behaviour is preserved by default -- ``enable_mixup`` is off unless a
        configuration sets it -- so this code reproduces the published runs
        rather than silently training a different model. See ``docs/errata.md``.
        """
        mix_lambda: torch.Tensor | None = None
        if getattr(self.config, "enable_mixup", False):
            alpha = float(getattr(self.config, "mixup_alpha", DEFAULT_MIXUP_ALPHA))
            mix_lambda = torch.from_numpy(
                get_mix_lambda(alpha, len(batch["waveform"]))
            ).to(self.device)

        pred, _ = self(batch["waveform"], mix_lambda)
        target = batch["target"].long()
        if self.config.loss_type == "weighted_clip_ce":
            loss = self.loss_func(pred, target, weight=self.loss_weight)
        else:
            loss = self.loss_func(pred, target)
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Reshuffle (and, if configured, rebalance) the training sampling queue."""
        if self.dataset is None:
            return
        self.dataset.generate_queue()

    # -- validation ------------------------------------------------------- #
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Accumulate validation logits and targets for the epoch-end metric."""
        pred, _ = self(batch["waveform"])
        self.validation_step_outputs.append((pred.detach(), batch["target"].detach()))

    def on_validation_epoch_end(self) -> None:
        """Log segment-level ``acc`` (monitored by ModelCheckpoint) and ``f1``."""
        pred = torch.cat([d[0] for d in self.validation_step_outputs], dim=0)
        target = torch.cat([d[1] for d in self.validation_step_outputs], dim=0)
        metric_dict, distributed = self._gather_and_evaluate(pred, target)
        world_scale = float(dist.get_world_size()) if distributed else 1.0
        self.log(
            "acc",
            metric_dict["acc"] * world_scale,
            on_epoch=True,
            prog_bar=True,
            sync_dist=distributed,
            batch_size=self.config.batch_size,
        )
        self.log(
            "f1",
            metric_dict["f1"] * world_scale,
            on_epoch=True,
            prog_bar=True,
            sync_dist=distributed,
            batch_size=self.config.batch_size,
        )
        if distributed:
            dist.barrier()
        self.validation_step_outputs.clear()

    # -- test ------------------------------------------------------------- #
    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Score one batch and hand it to :class:`PredictionWriter`.

        A single forward pass: the upstream 10-shift test-time augmentation was
        AudioSet-only (``dataset_type != 'audioset'`` forced ``shift_num = 1``),
        so no published number used it.
        """
        pred, _ = self(batch["waveform"])
        pred = pred.detach()
        target = batch["target"].detach()
        self.test_step_outputs.append((pred, target))
        return {
            LOGITS_KEY: pred.cpu(),
            TARGET_KEY: target.cpu(),
            AUDIO_NAME_KEY: list(batch["audio_name"]),
        }

    def on_test_epoch_end(self) -> None:
        """Log segment-level test metrics; patient-level pooling happens later."""
        pred = torch.cat([d[0] for d in self.test_step_outputs], dim=0).cpu().numpy()
        target = torch.cat([d[1] for d in self.test_step_outputs], dim=0).cpu().numpy()
        metric_dict = self.evaluate_metric(pred, target)
        print(f"segment-level test metrics: {metric_dict}", flush=True)
        self.log("acc", metric_dict["acc"], on_epoch=True, prog_bar=True)
        self.log("f1", metric_dict["f1"], on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()

    # -- optimisation ----------------------------------------------------- #
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        """AdamW plus the warm-up-then-decay schedule shared by all six models."""
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.02,
        )

        def lr_foo(epoch: int) -> float:
            """Return the learning-rate multiplier for ``epoch``."""
            if epoch < 3:
                # warm up lr
                return self.config.lr_rate[epoch]
            lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
            if lr_pos < -3:
                return max(self.config.lr_rate[0] * (0.98**epoch), 0.03)
            return self.config.lr_rate[lr_pos]

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]

    # -- internals -------------------------------------------------------- #
    def _gather_and_evaluate(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, float], bool]:
        """Evaluate on this rank, or on rank 0 over the gathered batch.

        Under DDP the metric is computed on rank 0 only and returned as zero
        elsewhere; the caller multiplies by the world size and logs with
        ``sync_dist=True``, whose mean then recovers the rank-0 value. This is
        the upstream scheme, kept so logged curves stay comparable.

        Whether the run is distributed is decided by ``torch.distributed``
        actually being initialised, not by the number of visible GPUs: the
        upstream check (``torch.cuda.device_count() > 1``) called
        ``dist.get_world_size()`` and crashed when a multi-GPU machine ran with
        ``--devices 1``.
        """
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed:
            return self.evaluate_metric(pred.cpu().numpy(), target.cpu().numpy()), False

        world_size = dist.get_world_size()
        gather_pred = [torch.zeros_like(pred) for _ in range(world_size)]
        gather_target = [torch.zeros_like(target) for _ in range(world_size)]
        dist.barrier()
        dist.all_gather(gather_pred, pred)
        dist.all_gather(gather_target, target)
        if dist.get_rank() != 0:
            return {"acc": 0.0, "f1": 0.0}, True
        metric_dict = self.evaluate_metric(
            torch.cat(gather_pred, dim=0).cpu().numpy(),
            torch.cat(gather_target, dim=0).cpu().numpy(),
        )
        print(f"rank 0 / {world_size}: {metric_dict}", flush=True)
        return metric_dict, True
