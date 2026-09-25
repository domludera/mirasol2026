"""
Abstract base class defining the interface all audio classification models must follow.

Any new model (CNN, ResNet, etc.) should subclass BaseAudioModel and implement
the forward() method, returning a dict with 'clipwise_output' and 'framewise_output'.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAudioModel(ABC, nn.Module):
    """Base class for all audio classification models.

    All models must accept raw waveform input and return a dictionary with:
        - 'clipwise_output': (batch_size, num_classes) tensor of clip-level predictions
        - 'framewise_output': (batch_size, num_frames, num_classes) tensor of frame-level predictions

    If a model does not produce frame-level predictions, it should return a
    zero tensor of appropriate shape for 'framewise_output'.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, mix_lambda=None, infer_mode=False
    ) -> dict:
        """Forward pass.

        Args:
            x: Input waveform tensor of shape (batch_size, num_samples).
            mix_lambda: Optional mixup lambda values for data augmentation.
            infer_mode: If True, handle variable-length audio input.

        Returns:
            dict with keys:
                'clipwise_output': (batch_size, num_classes)
                'framewise_output': (batch_size, num_frames, num_classes)
        """
        ...
