"""Forward-pass smoke tests for every registered architecture."""
import types

import pytest
import torch

from config import esc_config
from model.factory import SUPPORTED_MODELS, create_model

BATCH = 2


def _config(model_type):
    """Copy of esc_config with model_type overridden (leaves the module untouched)."""
    values = {k: v for k, v in vars(esc_config).items() if not k.startswith("__")}
    return types.SimpleNamespace(**{**values, "model_type": model_type})


@pytest.mark.slow
@pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
def test_forward_output_shapes(model_type):
    cfg = _config(model_type)
    model = create_model(cfg).eval()

    with torch.no_grad():
        out = model(torch.randn(BATCH, cfg.clip_samples))

    assert out["clipwise_output"].shape == (BATCH, cfg.classes_num)
    assert out["framewise_output"].shape[0] == BATCH
    assert out["framewise_output"].shape[-1] == cfg.classes_num


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        create_model(_config("not_a_model"))
