"""
Model factory — single entry point for creating audio classification models.

To add a new model:
    1. Create a new module in model/ (e.g. model/cnn.py)
    2. Ensure it subclasses BaseAudioModel from model.base
    3. Add a new entry to the _BUILDERS dict below
"""
from model.htsat import HTSAT_Swin_Transformer
from model.efficientnet import EfficientNetAudio
from model.timm_models import TimmAudioModel


def _build_htsat(config):
    """Build the HTSAT Swin Transformer model from config."""
    return HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head,
    )


def _build_efficientnet_b0(config):
    """Build the EfficientNet-B0 model from config."""
    return EfficientNetAudio(config)


def _build_fastvit(config):
    """Build the FastViT model from config."""
    model_name = getattr(config, "timm_model_name", "fastvit_t8.apple_in1k")
    return TimmAudioModel(config, model_name)


def _build_tinyvit(config):
    """Build the TinyViT model from config."""
    model_name = getattr(config, "timm_model_name", "tiny_vit_5m_224")
    return TimmAudioModel(config, model_name)


def _build_mobilenetv3(config):
    """Build the MobileNetV3-Small model from config."""
    return TimmAudioModel(config, "mobilenetv3_small_100.lamb_in1k")


def _build_resnet18(config):
    """Build the ResNet-18 model from config."""
    return TimmAudioModel(config, "resnet18.a1_in1k")


# ── Registry ────────────────────────────────────────────────────────────────
# Map model_type string → builder function.
# Each builder takes a config object and returns an nn.Module.
_BUILDERS = {
    "htsat": _build_htsat,
    "efficientnet_b0": _build_efficientnet_b0,
    "fastvit": _build_fastvit,
    "tinyvit": _build_tinyvit,
    "mobilenetv3": _build_mobilenetv3,
    "resnet18": _build_resnet18,
}

SUPPORTED_MODELS = tuple(_BUILDERS)


def create_model(config):
    """Instantiate a model based on config.model_type.

    Args:
        config: Configuration module (e.g. config.esc_config) that must have
                 a `model_type` attribute.

    Returns:
        An nn.Module ready to be passed to SEDWrapper.

    Raises:
        ValueError: If config.model_type is not a recognised model name.
    """
    model_type = getattr(config, "model_type", "htsat")

    if model_type not in _BUILDERS:
        available = ", ".join(sorted(_BUILDERS.keys()))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available models: {available}"
        )

    return _BUILDERS[model_type](config)
