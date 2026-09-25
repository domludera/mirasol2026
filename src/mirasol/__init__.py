"""MIRASOL 2026 - a low-cost AI stethoscope for heart-murmur screening.

Reference implementation accompanying:

    D. Ludera and M. Kersten-Oertel, "A Low-Cost AI Stethoscope for
    Resource-Constrained Heart Murmur Screening", MIRASOL @ MICCAI 2026.

The package is deliberately layered so that the analysis half of the paper can
be reproduced without PyTorch:

===========================  ====================================  ============
module                       paper artifact                        needs torch
===========================  ====================================  ============
``mirasol.paths``            every script's input/output locations  no
``mirasol.metrics``          Table 1, Table 2, Tables S1-S8, S10    no
``mirasol.signal.mel``       Fig. 1 pipeline, Table 4 model input   no
``mirasol.config``           training/export parameter provenance   no
``mirasol.data``             dataset preparation and folds          partly
``mirasol.models``           the six benchmarked architectures      yes
``mirasol.training``         five-fold cross-validation runs        yes
``mirasol.deployment``       ONNX export / quantization / edge      yes
===========================  ====================================  ============

Nothing in this package resolves a path from a hard-coded location; see
:mod:`mirasol.paths` for the single resolver used everywhere.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Release version of the reproduction package (independent of the paper).
__version__: str = "1.0.0"
