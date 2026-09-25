"""Configuration: read-only defaults plus one immutable override layer.

:mod:`mirasol.config.defaults` holds the published parameter values and nothing
else. :mod:`mirasol.config.resolve` builds an immutable
:class:`~mirasol.config.resolve.ResolvedConfig` from those defaults, an optional
YAML file and repeatable ``--set key=value`` flags, rejecting any key that is not
already defined so a typo fails at startup.

Together they replace the original scheme, in which nine scripts rewrote the
tracked defaults file in place with ``re.sub``: results depended on invocation
order, a crash mid-sweep left an arbitrary state, two folds could not run
concurrently, and a substitution that matched nothing changed nothing and was
never checked. Every run now records its effective values in
``runs/<exp_name>/resolved_config.json`` and no run writes to a tracked file.
"""

from __future__ import annotations

from mirasol.config.defaults import CLASS_NAMES, DEFAULTS, MODEL_TYPES
from mirasol.config.resolve import (
    ConfigError,
    ConfigValueError,
    ResolvedConfig,
    UnknownConfigKeyError,
    add_config_arguments,
    resolve_config,
    resolve_from_args,
)

__all__ = [
    "CLASS_NAMES",
    "ConfigError",
    "ConfigValueError",
    "DEFAULTS",
    "MODEL_TYPES",
    "ResolvedConfig",
    "UnknownConfigKeyError",
    "add_config_arguments",
    "resolve_config",
    "resolve_from_args",
]
