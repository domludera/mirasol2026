#!/usr/bin/env python3
"""Console entry point for quantization validation.

See :mod:`mirasol.deployment.validate`.  The historical filename is kept
because ``results/logs/int8_quantization.log`` and
``results/logs/static_fp16.log`` -- the only records of the Sec. 3.5 agreement
figures -- refer to ``validate_quantized.py`` by name.

    python deployment/validate_quantized.py --model-type tinyvit \\
        --checkpoint runs/.../l-epoch=143-acc=0.940.ckpt \\
        --onnx-model exports/tinyvit/model_int8_static.onnx --patient-level

is the same as ``python -m mirasol.deployment.validate ...``.
"""

from __future__ import annotations

import sys

from mirasol.deployment.validate import main

if __name__ == "__main__":
    sys.exit(main())
