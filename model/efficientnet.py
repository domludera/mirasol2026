"""
EfficientNet-B0 audio classification model.

This model processes raw audio waveforms into mel-spectrograms,
then passes them through a modified torchvision EfficientNet-B0.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseAudioModel
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

try:
    from src.utils import do_mixup
except ImportError:
    # Fallback if src.utils is not accessible or doesn't have do_mixup
    def do_mixup(x, mixup_lambda):
        out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
               x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
        return out


class EfficientNetAudio(BaseAudioModel):
    def __init__(self, config):
        """
        Args:
            config: Configuration object containing parameters for
                    spectrogram generation and model classes.
        """
        super().__init__()
        self.config = config

        # --- Audio Feature Extraction ---
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size, hop_length=config.hop_size,
            win_length=config.window_size, window=window, center=center,
            pad_mode=pad_mode, freeze_parameters=True
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate, n_fft=config.window_size,
            n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax,
            ref=ref, amin=amin, top_db=top_db, freeze_parameters=True
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=4, freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(config.mel_bins)

        # --- EfficientNet Backbone ---
        # Load a pre-trained EfficientNet-B0
        # `weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1` is the modern approach
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify the first conv layer to take 1 channel instead of 3
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Initialize the new conv layer's weights using the mean of the old weights (across RGB channels)
        with torch.no_grad():
            self.backbone.features[0][0].weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Modify the final classifier layer for the correct number of classes
        num_ftrs = self.backbone.classifier[1].in_features
        classes_num = getattr(config, 'classes_num', 2)
        self.backbone.classifier[1] = nn.Linear(num_ftrs, classes_num)

    def forward(self, x: torch.Tensor, mix_lambda=None, infer_mode=False) -> dict:
        """
        Forward pass from waveform to predictions.
        """
        # 1. Feature Extraction: Waveform -> Mel-Spectrogram
        x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)       # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)              # (batch_size, mel_bins, time_steps, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)              # (batch_size, 1, time_steps, mel_bins)

        if self.training:
            x = self.spec_augmenter(x)

        if self.training and mix_lambda is not None:
            x = do_mixup(x, mix_lambda)

        # 2. Reshape for CNN backbone
        # Current shape: (batch_size, 1, time_steps, mel_bins)
        # However, torchvision CNNs expect (batch_size, channels, height, width)
        # We can map: channels=1, height=mel_bins, width=time_steps
        x = x.transpose(2, 3)              # (batch_size, 1, mel_bins, time_steps)

        # 3. Backbone Forward Pass
        # Returns (batch_size, num_classes)
        clipwise_output = self.backbone(x)

        # 4. Prepare Output Dictionary
        # EfficientNet-B0 (as used here) only produces clipwise predictions.
        # Create a dummy framewise tensor to satisfy the BaseAudioModel interface.
        batch_size = x.size(0)
        time_steps = x.size(3)
        num_classes = clipwise_output.size(1)

        # Output dummy framewise of shape (batch, time_steps, num_classes)
        framewise_output = torch.zeros(
            (batch_size, time_steps, num_classes),
            device=x.device, dtype=x.dtype
        )

        output_dict = {
            'clipwise_output': clipwise_output,
            'framewise_output': framewise_output
        }

        return output_dict
