# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Dataset Collections
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
"""Torch dataset over the prepared CirCor fold file.

:class:`ESC_Dataset` (the name is inherited from HTS-AT's ESC-50 pipeline and is kept
because ``mirasol.config.defaults.dataset_type = "esc-50"`` and the released run
configs refer to it) reads the object array written by
:mod:`mirasol.data.prepare`: five folds, each a list of segment dicts with keys
``name``, ``target`` and ``waveform``.

Two behaviours matter for reproducing the paper and must not be "cleaned up":

1. **Self-concatenation.** ``__getitem__`` returns ``[waveform, waveform]``, so a 30 s
   / 120 000-sample window becomes 60 s / 240 000 samples and the log-mel front end
   produces ``(1, 2001, 64)``.  Every released ONNX graph has that 2001-frame time
   axis baked in.  :func:`mirasol.signal.mel.concatenate_with_self` is the torch-free
   mirror of this step.
2. **Class-balanced resampling.** With ``balanced_data = True`` - which every reported
   run used - the training queue is drawn with replacement so each epoch presents the
   three classes about equally often.  It is active *in addition to* the
   ``loss_weight = [4, 10, 1]`` cross-entropy weights; the paper mentions only the
   latter (see docs/errata.md).  Evaluation never resamples.

HTS-AT's AudioSet ``SEDDataset`` has been deleted: it was non-functional here, reading
six config keys (``enable_label_enhance``, ``enable_token_label``, ``class_filter``,
``crop_size``, ``class_map_path``, ``token_label_range``) that this project's
configuration has never defined.

Paper artifacts served: every trained model - Table 1, Table 3, Tables S7/S8/S10.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Segment(TypedDict):
    """One prepared audio segment, as written by :mod:`mirasol.data.prepare`."""

    name: str
    target: int
    waveform: np.ndarray


class Batch(TypedDict):
    """One item as handed to the Lightning module's ``training_step``/``test_step``."""

    audio_name: str
    waveform: np.ndarray
    real_len: int
    target: int


class DatasetConfig(Protocol):
    """Configuration keys :class:`ESC_Dataset` reads.

    ``esc_fold`` selects the held-out fold (0-4); ``classes_num`` sizes the
    per-class index used for balanced sampling; ``balanced_data`` switches that
    sampling on, and defaults to ``False`` if absent.
    """

    esc_fold: int
    classes_num: int
    balanced_data: bool


def _class_index(target: Any) -> int:
    """Return the class index of a segment target.

    Accepts both the scalar integer labels :mod:`mirasol.data.prepare` writes and the
    one-hot vectors HTS-AT's original pipeline used.

    Raises:
        TypeError: If ``target`` is neither a scalar nor an array-like.
    """
    if isinstance(target, (int, float, np.integer, np.floating)):
        return int(target)
    if isinstance(target, (np.ndarray, Sequence)):
        return int(np.argmax(target))
    raise TypeError(f"unsupported segment target of type {type(target).__name__}")


# For ESC dataset
class ESC_Dataset(Dataset):
    """Fold-aware dataset over the prepared 5-fold object array."""

    def __init__(
        self,
        dataset: Sequence[Sequence[Segment]],
        config: DatasetConfig,
        eval_mode: bool = False,
    ) -> None:
        """
        Args:
            dataset: The five folds loaded from the ``.npy`` file, i.e.
                ``np.load(path, allow_pickle=True)``.
            config: Provides ``esc_fold``, ``classes_num`` and ``balanced_data``.
            eval_mode: ``True`` yields exactly the held-out fold in stored order;
                ``False`` yields the other four folds, shuffled or class-balanced.

        Raises:
            IndexError: If ``config.esc_fold`` is not a valid fold index.
            ValueError: If the resulting split is empty.
        """
        self.config = config
        self.eval_mode = eval_mode

        num_folds = len(dataset)
        if not 0 <= config.esc_fold < num_folds:
            raise IndexError(
                f"esc_fold={config.esc_fold} is out of range for a {num_folds}-fold "
                "dataset; the published protocol uses folds 0-4."
            )

        if self.eval_mode:
            self.dataset: list[Segment] = list(dataset[config.esc_fold])
        else:
            train_segments: list[Segment] = []
            for i in range(num_folds):
                if i != config.esc_fold:
                    train_segments += list(dataset[i])
            self.dataset = train_segments

        self.total_size = len(self.dataset)
        if self.total_size == 0:
            raise ValueError(
                f"empty {'evaluation' if eval_mode else 'training'} split for "
                f"esc_fold={config.esc_fold}; the prepared .npy file has "
                f"{[len(fold) for fold in dataset]} segments per fold."
            )
        self.queue = [*range(self.total_size)]

        # Per-class segment indices, used for balanced sampling.
        self.idc: list[list[int]] = [[] for _ in range(config.classes_num)]
        for i, sample in enumerate(self.dataset):
            self.idc[_class_index(sample["target"])].append(i)

        print(f"{self.eval_mode} -- dataset size {self.total_size}")
        logger.info("total dataset size: %d", self.total_size)
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self) -> None:
        """Rebuild the sampling order for one training epoch.

        With ``balanced_data`` the queue is drawn *with replacement*, cycling through a
        shuffled class order and picking a random segment of each class, so a rare
        class is oversampled to roughly ``total_size / classes_num`` draws.  Without
        it the queue is a plain permutation.

        Raises:
            ValueError: If balanced sampling is requested but no class has any
                segment, which would otherwise spin forever.
        """
        if getattr(self.config, "balanced_data", False):
            populated = [indices for indices in self.idc if indices]
            if not populated:
                raise ValueError(
                    "balanced_data=True but no class has any segment; check "
                    "classes_num against the prepared .npy file's targets."
                )
            queue: list[int] = []
            while len(queue) < self.total_size:
                class_set = [*range(self.config.classes_num)]
                random.shuffle(class_set)
                for d in class_set:
                    if self.idc[d]:
                        queue.append(self.idc[d][random.randint(0, len(self.idc[d]) - 1)])
            self.queue = queue[: self.total_size]
        else:
            self.queue = [*range(self.total_size)]
            random.shuffle(self.queue)
        logger.info("queue regenerated:%s", self.queue[-5:])

    def __getitem__(self, index: int) -> Batch:
        """Load the waveform and target of one segment.

        Args:
            index: Position in the epoch's sampling queue.

        Returns:
            A :class:`Batch`.  ``waveform`` is the segment concatenated with itself
            (30 s -> 60 s -> 2001 mel frames) and ``real_len`` records the resulting
            length in samples.
        """
        p = self.queue[index]
        sample = self.dataset[p]
        waveform = sample["waveform"]
        return {
            "audio_name": sample["name"],
            "waveform": np.concatenate((waveform, waveform)),
            "real_len": len(waveform) * 2,
            "target": sample["target"],
        }

    def __len__(self) -> int:
        return self.total_size
