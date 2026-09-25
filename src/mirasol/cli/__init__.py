# SPDX-License-Identifier: MIT
"""Console entry points installed with the package.

* ``mirasol-train`` -> :mod:`mirasol.cli.train`, which trains or evaluates one
  architecture on one fold and writes ``runs/<exp>/predictions.csv``.
* ``mirasol-count-params`` -> :mod:`mirasol.cli.count_params`, which regenerates
  Table 1's parameter-count column headers.

Both modules are also runnable as ``python -m mirasol.cli.<name>``, which is how
``experiments/run_experiment.py`` invokes the trainer. Nothing is imported here:
each entry point pulls in torch only when it actually runs.
"""

from __future__ import annotations

__all__ = ["count_params", "train"]
