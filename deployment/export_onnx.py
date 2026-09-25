#!/usr/bin/env python3
"""Console entry point for ONNX export -- see :mod:`mirasol.deployment.export`.

The historical filename is kept because the shipped provenance logs
(``results/logs/int8_quantization.log``, ``results/logs/static_fp16.log``) refer
to ``export_onnx.py`` by name.  All logic lives in the package:

    python deployment/export_onnx.py --model-type efficientnet_b0 \\
        --checkpoint runs/.../l-epoch=142-acc=0.855.ckpt --static --fp16

is the same as

    python -m mirasol.deployment.export ...

Static INT8 defaults to per-channel weight scales; ``--per-tensor`` reproduces
the graphs published with the paper.  See ``docs/quantization.md``.
"""

from __future__ import annotations

import sys

from mirasol.deployment.export import main

if __name__ == "__main__":
    sys.exit(main())
