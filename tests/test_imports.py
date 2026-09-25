"""Entry points must import without anything outside the repository."""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "main_moody", "export_onnx", "src.sed_model", "scripts.experiment_utils",
    "scripts.run_5fold", "scripts.run_ablation", "scripts.export_all",
])
def test_entry_point_imports(module):
    importlib.import_module(module)
