"""Build an immutable configuration from defaults + YAML + ``--set`` overrides.

This module is the *only* parameter channel in the release. It exists because
the original workflow had nine scripts rewrite the tracked defaults file in
place with ``re.sub``, with four consequences that all silently corrupted
results:

* each script patched a different subset of keys, so a run inherited whatever
  the previous one left behind (Table S7's driver never set ``classes_num`` or
  ``loss_weight`` at all, and would build a three-output head on two-class data);
* a crash mid-sweep left the tracked file in an arbitrary state;
* two folds could not run concurrently;
* ``re.sub`` returns the input unchanged when the pattern misses, and no script
  checked the substitution count.

The replacement is a frozen :class:`ResolvedConfig`. Overrides must name a key
that already exists in :mod:`mirasol.config.defaults`, so a typo raises at
startup instead of being ignored; values are coerced to the type of the default,
so ``--set classes_num=two`` fails immediately; and a small set of cross-key
invariants is checked, so the specific inconsistency behind the Table S7 blocker
(``len(loss_weight) != classes_num``) cannot happen at all.

Nothing here writes to a version-controlled path. The provenance record is
``runs/<exp_name>/resolved_config.json``, written by
:meth:`ResolvedConfig.write`, and :meth:`mirasol.paths.Paths.assert_writable`
refuses any target inside the tracked tree.

Paper artifacts served: every run behind Table 1, Table S7, Table S8, Table S10
and the class-weight sweep of Sec. 3.6 is fully described by its
``resolved_config.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from mirasol import __version__
from mirasol.config import defaults as _defaults
from mirasol.paths import Paths, add_path_arguments, paths_from_args, resolve_paths
from mirasol.signal.mel import MelConfig, mel_config_from_mapping

__all__ = [
    "ConfigError",
    "ConfigValueError",
    "ResolvedConfig",
    "UnknownConfigKeyError",
    "add_config_arguments",
    "load_yaml_overrides",
    "parse_set_overrides",
    "resolve_config",
    "resolve_from_args",
]

#: Filename of the per-run provenance record.
RESOLVED_CONFIG_FILENAME: Final[str] = "resolved_config.json"

#: Keys computed from other keys. They may appear in an override only if the
#: value agrees with the derivation, so an inconsistent pair cannot slip through.
_DERIVED_KEYS: Final[tuple[str, ...]] = ("clip_samples",)

_TRUE_WORDS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_NULL_WORDS: Final[frozenset[str]] = frozenset({"none", "null", ""})

_ALLOWED_ACCELERATORS: Final[frozenset[str]] = frozenset(
    {"auto", "cpu", "gpu", "cuda", "mps"}
)


class ConfigError(Exception):
    """Base class for configuration failures."""


class UnknownConfigKeyError(ConfigError):
    """An override names a parameter that does not exist."""


class ConfigValueError(ConfigError):
    """An override value cannot be used for the parameter it names."""


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def _coerce_scalar(key: str, raw: Any, reference: Any) -> Any:
    """Coerce *raw* to the type of *reference*, or raise :class:`ConfigValueError`."""
    if reference is None:
        # ``None`` defaults are optional paths/names: accept a string or an
        # explicit null, and nothing else.
        if raw is None:
            return None
        text = str(raw).strip()
        return None if text.lower() in _NULL_WORDS else text

    if isinstance(reference, bool):
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        raise ConfigValueError(
            f"{key}: expected a boolean, got {raw!r} (use true/false)"
        )

    if isinstance(reference, int):
        try:
            return int(str(raw).strip())
        except ValueError:
            raise ConfigValueError(f"{key}: expected an integer, got {raw!r}") from None

    if isinstance(reference, float):
        try:
            return float(str(raw).strip())
        except ValueError:
            raise ConfigValueError(f"{key}: expected a number, got {raw!r}") from None

    if isinstance(reference, str):
        return str(raw)

    raise ConfigValueError(
        f"{key}: no coercion rule for default of type {type(reference).__name__}"
    )


def _coerce_sequence(key: str, raw: Any, reference: tuple[Any, ...]) -> tuple[Any, ...]:
    """Coerce *raw* to a tuple whose elements match *reference*'s element type."""
    if isinstance(raw, (list, tuple)):
        items: Sequence[Any] = tuple(raw)
    else:
        text = str(raw).strip()
        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise ConfigValueError(
                    f"{key}: cannot parse {raw!r} as a list; write e.g. [4.0,10.0,1.0]"
                ) from None
            if not isinstance(parsed, (list, tuple)):
                raise ConfigValueError(f"{key}: expected a list, got {raw!r}")
            items = tuple(parsed)
        else:
            items = tuple(part for part in text.split(",") if part.strip() != "")
    if not items:
        raise ConfigValueError(f"{key}: expected at least one value, got {raw!r}")
    element_reference = reference[0] if reference else ""
    return tuple(
        _coerce_scalar(f"{key}[{index}]", item, element_reference)
        for index, item in enumerate(items)
    )


def _coerce(key: str, raw: Any, reference: Any) -> Any:
    """Coerce one override value against the type of its default."""
    if isinstance(reference, tuple):
        return _coerce_sequence(key, raw, reference)
    return _coerce_scalar(key, raw, reference)


def parse_set_overrides(items: Iterable[str]) -> Mapping[str, str]:
    """Parse repeatable ``--set key=value`` arguments into a mapping.

    A repeated key is an error rather than a last-wins silent override, because
    the original bug class was precisely "two places set the same parameter and
    only one of them was noticed".
    """
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigValueError(
                f"--set expects key=value, got {item!r}"
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ConfigValueError(f"--set expects a non-empty key, got {item!r}")
        if key in parsed:
            raise ConfigValueError(
                f"--set {key} given twice ({parsed[key]!r} then {value!r}); "
                "specify each parameter once"
            )
        parsed[key] = value
    return MappingProxyType(parsed)


def load_yaml_overrides(path: Path | str) -> Mapping[str, Any]:
    """Read a flat YAML mapping of overrides.

    Nested mappings are rejected: the parameter space is flat, and a nested file
    would silently contribute nothing.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ConfigError(
            "reading a YAML config requires PyYAML; install the package with "
            "`pip install -e .` or pass parameters with --set instead"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return MappingProxyType({})
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"{path}: expected a mapping of parameter names to values, got "
            f"{type(loaded).__name__}"
        )
    nested = sorted(
        key for key, value in loaded.items() if isinstance(value, (dict, Mapping))
    )
    if nested:
        raise ConfigError(
            f"{path}: the parameter space is flat, but {nested} map to sub-mappings"
        )
    return MappingProxyType({str(key): value for key, value in loaded.items()})


# ---------------------------------------------------------------------------
# ResolvedConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """An immutable, fully validated parameter set with its provenance.

    Parameters are readable as attributes (``config.classes_num``) so that code
    ported from the original tree keeps working, and as a mapping
    (``config["classes_num"]``) for generic access. There is no setter; use
    :meth:`with_overrides` to derive a variant.
    """

    values: Mapping[str, Any]
    sources: Mapping[str, str]
    paths: Paths = field(default_factory=resolve_paths)

    # -- mapping-ish access ----------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            values: Mapping[str, Any] = object.__getattribute__(self, "values")
        except AttributeError:  # pragma: no cover - during unpickling only
            raise AttributeError(name) from None
        if name in values:
            return values[name]
        raise AttributeError(
            f"{name!r} is not a configuration parameter. Known parameters: "
            f"{', '.join(sorted(values))}"
        )

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def keys(self) -> Iterable[str]:
        """Parameter names, sorted."""
        return self.values.keys()

    def get(self, key: str, default: Any = None) -> Any:
        """Parameter value, or *default* if the parameter does not exist."""
        return self.values.get(key, default)

    # -- derived views ----------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """``runs/<exp_name>``: where this run's outputs and record belong."""
        return self.paths.run_dir(self.values["exp_name"])

    @property
    def dataset_npy(self) -> Path:
        """Absolute path of the prepared fold array this run reads."""
        return self.paths.circor_root / self.values["dataset_npy_name"]

    @property
    def mel_config(self) -> MelConfig:
        """Spectrogram parameters as consumed by :mod:`mirasol.signal.mel`."""
        return mel_config_from_mapping(self.values)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Label names for this run's ``classes_num``."""
        if self.values["classes_num"] == 3:
            return _defaults.CLASS_NAMES
        return ("Present or Unknown", "Absent")

    # -- derivation and serialisation -------------------------------------

    def with_overrides(
        self, overrides: Mapping[str, Any], *, source: str = "runtime"
    ) -> ResolvedConfig:
        """Return a new :class:`ResolvedConfig` with *overrides* applied."""
        merged, provenance = _apply_overrides(
            dict(self.values), dict(self.sources), overrides, source
        )
        return _finalise(merged, provenance, self.paths)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the parameters (tuples become lists)."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in sorted(self.values.items())
        }

    def provenance(self) -> dict[str, Any]:
        """The full record written to ``resolved_config.json``."""
        return {
            "mirasol_version": __version__,
            "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": self.as_dict(),
            "sources": dict(sorted(self.sources.items())),
            "paths": dict(self.paths.as_dict()),
        }

    def write(self, directory: Path | None = None) -> Path:
        """Write ``resolved_config.json`` and return its path.

        Defaults to :attr:`run_dir`. Refuses to write anywhere under version
        control, which is the guarantee that replaced in-place config rewriting.
        """
        target_dir = self.run_dir if directory is None else Path(directory)
        self.paths.ensure(target_dir)
        target = target_dir / RESOLVED_CONFIG_FILENAME
        with target.open("w") as handle:
            json.dump(self.provenance(), handle, indent=2, sort_keys=False)
            handle.write("\n")
        return target


# ---------------------------------------------------------------------------
# Resolution pipeline
# ---------------------------------------------------------------------------


def _apply_overrides(
    values: dict[str, Any],
    sources: dict[str, str],
    overrides: Mapping[str, Any],
    source: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate and apply *overrides*, returning new dicts (inputs untouched)."""
    merged = dict(values)
    provenance = dict(sources)
    unknown = sorted(set(overrides) - set(_defaults.DEFAULTS))
    if unknown:
        raise UnknownConfigKeyError(
            f"unknown configuration key(s) {unknown} from {source}. "
            "Overrides must name a parameter that already exists in "
            "mirasol.config.defaults; this check exists so a misspelt parameter "
            f"cannot be silently ignored. Known parameters: "
            f"{', '.join(sorted(_defaults.DEFAULTS))}"
        )
    for key, raw in overrides.items():
        merged[key] = _coerce(key, raw, _defaults.DEFAULTS[key])
        provenance[key] = source
    return merged, provenance


def _finalise(
    values: dict[str, Any], sources: dict[str, str], paths: Paths
) -> ResolvedConfig:
    """Recompute derived keys, validate invariants, and freeze."""
    derived = _resolve_derived(values, sources)
    _validate(derived)
    return ResolvedConfig(
        values=MappingProxyType(dict(sorted(derived.items()))),
        sources=MappingProxyType(dict(sorted(sources.items()))),
        paths=paths,
    )


def _resolve_derived(
    values: dict[str, Any], sources: dict[str, str]
) -> dict[str, Any]:
    """Recompute derived keys; an explicit but inconsistent value is an error."""
    updated = dict(values)
    expected = {"clip_samples": values["sample_rate"] * values["clip_seconds"]}
    for key in _DERIVED_KEYS:
        computed = expected[key]
        if sources.get(key, "default") == "default":
            updated[key] = computed
        elif values[key] != computed:
            raise ConfigValueError(
                f"{key}={values[key]} contradicts sample_rate * clip_seconds = "
                f"{computed}; {key} is derived, so override sample_rate or "
                "clip_seconds instead of setting it directly"
            )
    return updated


def _validate(values: Mapping[str, Any]) -> None:
    """Check the cross-key invariants that silently broke published runs."""
    model_type = values["model_type"]
    if model_type not in _defaults.MODEL_TYPES:
        raise ConfigValueError(
            f"model_type={model_type!r} is not one of {list(_defaults.MODEL_TYPES)}"
        )

    classes_num = values["classes_num"]
    if classes_num not in (2, 3):
        raise ConfigValueError(
            f"classes_num={classes_num}; this release supports the three-class "
            "protocol (Present/Unknown/Absent) and the two-class configuration "
            "of Table S7"
        )

    loss_weight = values["loss_weight"]
    if values["loss_type"] == "weighted_clip_ce" and len(loss_weight) != classes_num:
        raise ConfigValueError(
            f"loss_weight has {len(loss_weight)} entries but classes_num="
            f"{classes_num}. A mismatch here builds a head whose extra logit is "
            "never trained - the defect that made Table S7 unreproducible. Set "
            "both together (3-class: [4.0,10.0,1.0]; 2-class: [4.0,1.0])."
        )
    if any(weight <= 0 for weight in loss_weight):
        raise ConfigValueError(
            f"loss_weight entries must be positive, got {list(loss_weight)}"
        )

    esc_fold = values["esc_fold"]
    if not 0 <= esc_fold <= 4:
        raise ConfigValueError(
            f"esc_fold={esc_fold}; the published protocol has five patient-level "
            "folds indexed 0-4 (see docs/splits.md)"
        )

    if values["dataset_type"] != "esc-50":
        raise ConfigValueError(
            f"dataset_type={values['dataset_type']!r}; only the ESC-style "
            "fold-list loader is part of this release"
        )

    accelerator = values["accelerator"]
    if accelerator not in _ALLOWED_ACCELERATORS:
        raise ConfigValueError(
            f"accelerator={accelerator!r} is not one of "
            f"{sorted(_ALLOWED_ACCELERATORS)}"
        )

    exp_name = values["exp_name"]
    if not exp_name or "/" in exp_name or exp_name in {".", ".."}:
        raise ConfigValueError(
            f"exp_name={exp_name!r} must be a single non-empty directory name; it "
            "names runs/<exp_name>/"
        )

    for key in ("batch_size", "max_epoch", "save_top_k", "val_check_interval"):
        if values[key] < 1:
            raise ConfigValueError(f"{key}={values[key]} must be at least 1")
    if values["num_workers"] < 0:
        raise ConfigValueError(f"num_workers={values['num_workers']} must be >= 0")
    if values["learning_rate"] <= 0:
        raise ConfigValueError(
            f"learning_rate={values['learning_rate']} must be positive"
        )

    # Constructing the MelConfig validates the spectrogram parameters (window
    # parity, band inside Nyquist) before a run starts rather than at the first
    # forward pass.
    mel_config_from_mapping(values)


def resolve_config(
    *,
    config_file: Path | str | None = None,
    set_overrides: Iterable[str] = (),
    overrides: Mapping[str, Any] | None = None,
    paths: Paths | None = None,
) -> ResolvedConfig:
    """Resolve defaults + YAML + ``--set`` + programmatic overrides, in that order.

    Args:
        config_file: optional flat YAML file of parameter overrides.
        set_overrides: repeatable ``"key=value"`` strings from the command line.
        overrides: already-typed overrides, applied last. Drivers use this to
            write a *complete* parameter set for an experiment (task, model,
            fold, dataset, class count, loss weights) instead of relying on
            inheritance.
        paths: filesystem locations; defaults to :func:`mirasol.paths.resolve_paths`.

    Raises:
        UnknownConfigKeyError: an override names a parameter that does not exist.
        ConfigValueError: a value has the wrong type or breaks an invariant.
    """
    values = dict(_defaults.DEFAULTS)
    sources = {key: "default" for key in values}
    resolved_paths = resolve_paths() if paths is None else paths

    if config_file is not None:
        values, sources = _apply_overrides(
            values,
            sources,
            load_yaml_overrides(config_file),
            f"yaml:{Path(config_file).name}",
        )
    if set_overrides:
        values, sources = _apply_overrides(
            values, sources, parse_set_overrides(set_overrides), "--set"
        )
    if overrides:
        values, sources = _apply_overrides(values, sources, overrides, "driver")
    return _finalise(values, sources, resolved_paths)


def add_config_arguments(
    parser: argparse.ArgumentParser, *, include_paths: bool = True
) -> argparse.ArgumentParser:
    """Add ``--config``/``--set`` (and optionally the path flags) to *parser*."""
    group = parser.add_argument_group(
        "configuration",
        "Defaults come from mirasol.config.defaults and are never modified on "
        "disk; --config and --set layer on top.",
    )
    group.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="flat YAML file of parameter overrides",
    )
    group.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one parameter; repeatable. Unknown keys are rejected.",
    )
    if include_paths:
        add_path_arguments(parser)
    return parser


def resolve_from_args(
    args: argparse.Namespace, *, overrides: Mapping[str, Any] | None = None
) -> ResolvedConfig:
    """Resolve a configuration from a parsed namespace.

    Works with any parser passed through :func:`add_config_arguments`; missing
    destinations are treated as absent so a script may expose a subset of the
    flags without special-casing.
    """
    return resolve_config(
        config_file=getattr(args, "config", None),
        set_overrides=getattr(args, "set_overrides", ()) or (),
        overrides=overrides,
        paths=paths_from_args(args),
    )
