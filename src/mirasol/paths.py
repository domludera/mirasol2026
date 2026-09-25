"""Single source of truth for every filesystem location used by this release.

Nothing else in the repository may embed a directory name. Each location is
resolved with a fixed precedence:

    explicit argument (CLI flag)  >  environment variable  >  repo-relative default

and the repository root is derived from this file's own location, so a clone
works from any working directory and no absolute path is ever committed.

Inputs versus outputs
---------------------
Two kinds of location are handled differently on purpose:

*Inputs* that this repository does not (and legally may not) redistribute -
the CirCor DigiScope metadata CSV and the ~560 MB WAV corpus - are never
created. Asking for one that is absent raises :class:`MissingInputError` with a
single actionable message naming the PhysioNet record, because a reproducer's
most likely failure is simply not having downloaded the dataset yet.

*Outputs* - ``runs/``, ``exports/`` and anything under ``results/`` that a
script generates - are created on demand via ``ensure=True``.

Paper artifacts served
----------------------
Every table and figure script (``analysis/table1_fivefold.py``,
``analysis/table2_clinical.py``, ``analysis/supp_tables.py``,
``analysis/figures.py``, ``analysis/table4_ondevice.py``) and every training,
export and benchmark entry point resolves its locations here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

__all__ = [
    "BENCHMARKS_DIR",
    "CIRCOR_AUDIO",
    "CIRCOR_META",
    "CIRCOR_ROOT",
    "EXPORTS_DIR",
    "FOLDS_CSV",
    "LOGS_DIR",
    "MissingInputError",
    "PHYSIONET_URL",
    "PREDS_DIR",
    "Paths",
    "REPO_ROOT",
    "RESULTS_DIR",
    "RUNS_DIR",
    "SPLITS_DIR",
    "add_path_arguments",
    "default_paths",
    "paths_from_args",
    "resolve_paths",
]

#: The canonical citation target for the dataset. Named in every error message
#: about a missing CirCor input so the reader never has to search for it.
PHYSIONET_URL: Final[str] = "https://physionet.org/content/circor-heart-sound/"

#: Repository root. ``paths.py`` lives at ``<root>/src/mirasol/paths.py``.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Environment variables and repo-relative defaults
# ---------------------------------------------------------------------------

#: Primary environment variable per location, followed by accepted aliases. The
#: ``MIRASOL_``-prefixed forms exist so a shell can namespace every override;
#: the unprefixed CirCor names match what the PhysioNet download instructions in
#: ``data/README.md`` tell the reader to export.
ENV_CIRCOR_ROOT: Final[str] = "CIRCOR_ROOT"
ENV_CIRCOR_META: Final[str] = "CIRCOR_META"
ENV_CIRCOR_AUDIO: Final[str] = "CIRCOR_AUDIO"
ENV_PREDS_DIR: Final[str] = "MIRASOL_PREDS_DIR"
ENV_RESULTS_DIR: Final[str] = "MIRASOL_RESULTS_DIR"
ENV_RUNS_DIR: Final[str] = "MIRASOL_RUNS_DIR"
ENV_EXPORTS_DIR: Final[str] = "MIRASOL_EXPORTS_DIR"
ENV_SPLITS_DIR: Final[str] = "MIRASOL_SPLITS_DIR"
ENV_REPO_ROOT: Final[str] = "MIRASOL_REPO_ROOT"

_ENV_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    ENV_REPO_ROOT: (ENV_REPO_ROOT,),
    ENV_CIRCOR_ROOT: (ENV_CIRCOR_ROOT, "MIRASOL_CIRCOR_ROOT"),
    ENV_CIRCOR_META: (ENV_CIRCOR_META, "MIRASOL_CIRCOR_META"),
    ENV_CIRCOR_AUDIO: (ENV_CIRCOR_AUDIO, "MIRASOL_CIRCOR_AUDIO", "MIRASOL_AUDIO_DIR"),
    ENV_PREDS_DIR: (ENV_PREDS_DIR, "PREDS_DIR"),
    ENV_RESULTS_DIR: (ENV_RESULTS_DIR,),
    ENV_RUNS_DIR: (ENV_RUNS_DIR,),
    ENV_EXPORTS_DIR: (ENV_EXPORTS_DIR,),
    ENV_SPLITS_DIR: (ENV_SPLITS_DIR,),
}

#: Repo-relative fallbacks. ``CIRCOR_META``/``CIRCOR_AUDIO`` are relative to the
#: resolved CirCor root rather than to the repository root, because they are
#: members of the downloaded record.
DEFAULT_CIRCOR_ROOT: Final[str] = "data/circor"
DEFAULT_CIRCOR_META_NAME: Final[str] = "training_data.csv"
DEFAULT_CIRCOR_AUDIO_NAME: Final[str] = "training_data"
DEFAULT_PREDS_DIR: Final[str] = "results/predictions"
DEFAULT_RESULTS_DIR: Final[str] = "results"
DEFAULT_RUNS_DIR: Final[str] = "runs"
DEFAULT_EXPORTS_DIR: Final[str] = "exports"
DEFAULT_SPLITS_DIR: Final[str] = "data/splits"

#: Repo-relative directories that hold version-controlled source or evidence.
#: No generated file may be written into any of them; see
#: :meth:`Paths.assert_writable`. ``results/`` as a whole is deliberately *not*
#: listed - scripts add new benchmark JSON there - but the committed prediction
#: dumps and fold table are.
VERSION_CONTROLLED_PREFIXES: Final[tuple[str, ...]] = (
    "src",
    "analysis",
    "experiments",
    "deployment",
    "docs",
    "hardware",
    "tests",
    ".github",
    "data/splits",
    "results/predictions",
    "results/logs",
)


class MissingInputError(FileNotFoundError):
    """A required input that this repository does not redistribute is absent."""


def _env_path(variable: str) -> Path | None:
    """Return the path from *variable* or one of its aliases, else ``None``.

    Aliases are tried in declaration order and the first non-empty value wins, so
    ``CIRCOR_ROOT`` takes precedence over ``MIRASOL_CIRCOR_ROOT``.
    """
    for name in _ENV_ALIASES.get(variable, (variable,)):
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw).expanduser()
    return None


def _first(*candidates: Path | str | None) -> Path | None:
    """Return the first non-``None`` candidate as a :class:`~pathlib.Path`."""
    for candidate in candidates:
        if candidate is not None:
            return Path(candidate).expanduser()
    return None


@dataclass(frozen=True)
class Paths:
    """An immutable, fully resolved set of locations.

    Build one with :func:`resolve_paths`, :func:`default_paths` or
    :func:`paths_from_args`; never mutate it. Use :meth:`replace_with` to
    derive a variant.
    """

    repo_root: Path
    circor_root: Path
    circor_meta: Path
    circor_audio: Path
    preds_dir: Path
    results_dir: Path
    runs_dir: Path
    exports_dir: Path
    splits_dir: Path

    # -- derived locations ------------------------------------------------

    @property
    def folds_csv(self) -> Path:
        """``data/splits/folds.csv``: the shipped 941-row patient/fold table."""
        return self.splits_dir / "folds.csv"

    @property
    def benchmarks_dir(self) -> Path:
        """``results/benchmarks``: the on-device latency/memory JSON records."""
        return self.results_dir / "benchmarks"

    @property
    def logs_dir(self) -> Path:
        """``results/logs``: the shipped quantization and QAT run logs."""
        return self.results_dir / "logs"

    def run_dir(self, exp_name: str) -> Path:
        """Return ``runs/<exp_name>``, the provenance directory for one run."""
        if not exp_name or "/" in exp_name or exp_name in {".", ".."}:
            raise ValueError(
                f"exp_name must be a single non-empty directory name, got {exp_name!r}"
            )
        return self.runs_dir / exp_name

    def export_dir(self, model_type: str) -> Path:
        """Return ``exports/<model_type>``, where ONNX graphs are written."""
        if not model_type or "/" in model_type:
            raise ValueError(
                f"model_type must be a single non-empty directory name, got {model_type!r}"
            )
        return self.exports_dir / model_type

    def prediction_csv(self, exp_name: str) -> Path:
        """Return the out-of-fold prediction dump for *exp_name*.

        The shipped files keep the historical ``<exp_name>.data.csv`` naming so
        that ``results/predictions/`` stays byte-identical to the artifacts the
        published tables were computed from.
        """
        return self.preds_dir / f"{exp_name}.data.csv"

    # -- required inputs --------------------------------------------------

    def require_circor_meta(self) -> Path:
        """Return the CirCor metadata CSV, or raise one actionable error."""
        return self._require(
            self.circor_meta,
            kind="file",
            what="the CirCor DigiScope metadata CSV (training_data.csv, 114 KB)",
            flag="--circor-meta",
            variable=ENV_CIRCOR_META,
        )

    def require_circor_audio(self) -> Path:
        """Return the CirCor WAV directory, or raise one actionable error."""
        return self._require(
            self.circor_audio,
            kind="directory",
            what="the CirCor DigiScope recordings directory (training_data/, ~560 MB)",
            flag="--audio-dir",
            variable=ENV_CIRCOR_AUDIO,
        )

    def require_preds_dir(self) -> Path:
        """Return the out-of-fold prediction directory, or raise."""
        return self._require(
            self.preds_dir,
            kind="directory",
            what="the out-of-fold prediction dumps (results/predictions/*.data.csv)",
            flag="--preds-dir",
            variable=ENV_PREDS_DIR,
            circor_download=False,
        )

    def require_folds_csv(self) -> Path:
        """Return ``data/splits/folds.csv``, or raise."""
        return self._require(
            self.folds_csv,
            kind="file",
            what="the patient/fold table (data/splits/folds.csv, 941 rows)",
            flag="--splits-dir",
            variable=ENV_SPLITS_DIR,
            circor_download=False,
        )

    # -- outputs ----------------------------------------------------------

    def ensure(self, directory: Path) -> Path:
        """Create *directory* (and parents) if needed and return it.

        Refuses to create anything inside a version-controlled directory, which
        is what stops a run from dirtying tracked files (release blocker 10).
        """
        self.assert_writable(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def assert_writable(self, target: Path) -> None:
        """Raise :class:`PermissionError` if *target* is version-controlled.

        Paths outside the repository are always allowed: a user may legitimately
        direct output to a scratch disk.
        """
        resolved = Path(target).expanduser()
        if not resolved.is_absolute():
            resolved = (self.repo_root / resolved).resolve()
        try:
            relative = resolved.relative_to(self.repo_root.resolve())
        except ValueError:
            return
        as_posix = relative.as_posix()
        for prefix in VERSION_CONTROLLED_PREFIXES:
            if as_posix == prefix or as_posix.startswith(prefix + "/"):
                raise PermissionError(
                    f"refusing to write {as_posix!r}: {prefix}/ is version-controlled. "
                    "Generated files belong under runs/, exports/ or an explicit "
                    "--out directory."
                )

    def replace_with(self, **overrides: Path) -> Paths:
        """Return a new :class:`Paths` with *overrides* applied."""
        return replace(self, **overrides)

    def as_dict(self) -> Mapping[str, str]:
        """Return a JSON-serialisable, repo-relative-where-possible mapping."""
        root = self.repo_root.resolve()
        out: dict[str, str] = {"repo_root": str(root)}
        for field_name in (
            "circor_root",
            "circor_meta",
            "circor_audio",
            "preds_dir",
            "results_dir",
            "runs_dir",
            "exports_dir",
            "splits_dir",
        ):
            value: Path = getattr(self, field_name)
            try:
                out[field_name] = value.resolve().relative_to(root).as_posix()
            except ValueError:
                out[field_name] = str(value)
        return out

    # -- internals --------------------------------------------------------

    def _require(
        self,
        path: Path,
        *,
        kind: str,
        what: str,
        flag: str,
        variable: str,
        circor_download: bool = True,
    ) -> Path:
        exists = path.is_dir() if kind == "directory" else path.is_file()
        if exists:
            return path
        lines = [
            f"Could not find {what}.",
            f"  looked for {kind}: {path}",
            f"  override with {flag} <path> or the {variable} environment variable",
        ]
        if circor_download:
            lines.append(
                "  CirCor DigiScope is not redistributed with this repository; "
                f"download it from {PHYSIONET_URL} "
                "(the clinical-endpoint analysis needs only training_data.csv, "
                "not the audio) and see data/README.md for the expected layout."
            )
        else:
            lines.append(
                "  this file ships with the repository; if it is missing, the clone "
                "is incomplete - re-clone from https://github.com/domludera/mirasol2026"
            )
        raise MissingInputError("\n".join(lines))


def resolve_paths(
    *,
    repo_root: Path | str | None = None,
    circor_root: Path | str | None = None,
    circor_meta: Path | str | None = None,
    circor_audio: Path | str | None = None,
    preds_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    runs_dir: Path | str | None = None,
    exports_dir: Path | str | None = None,
    splits_dir: Path | str | None = None,
) -> Paths:
    """Resolve every location once, honouring explicit > env > default.

    All arguments are optional; passing one is equivalent to a CLI flag and
    takes precedence over the corresponding environment variable. Relative
    arguments are interpreted relative to the current working directory (they
    came from a user's shell), which is why they are expanded here rather than
    joined to ``repo_root``.
    """
    root = _first(repo_root, _env_path(ENV_REPO_ROOT), REPO_ROOT)
    assert root is not None  # REPO_ROOT is a non-None fallback
    root = root.resolve()

    resolved_circor_root = _first(
        circor_root, _env_path(ENV_CIRCOR_ROOT), root / DEFAULT_CIRCOR_ROOT
    )
    assert resolved_circor_root is not None

    def _under_root(explicit: Path | str | None, variable: str, default: str) -> Path:
        value = _first(explicit, _env_path(variable), root / default)
        assert value is not None
        return value

    resolved_meta = _first(
        circor_meta,
        _env_path(ENV_CIRCOR_META),
        resolved_circor_root / DEFAULT_CIRCOR_META_NAME,
    )
    resolved_audio = _first(
        circor_audio,
        _env_path(ENV_CIRCOR_AUDIO),
        resolved_circor_root / DEFAULT_CIRCOR_AUDIO_NAME,
    )
    assert resolved_meta is not None and resolved_audio is not None

    return Paths(
        repo_root=root,
        circor_root=resolved_circor_root,
        circor_meta=resolved_meta,
        circor_audio=resolved_audio,
        preds_dir=_under_root(preds_dir, ENV_PREDS_DIR, DEFAULT_PREDS_DIR),
        results_dir=_under_root(results_dir, ENV_RESULTS_DIR, DEFAULT_RESULTS_DIR),
        runs_dir=_under_root(runs_dir, ENV_RUNS_DIR, DEFAULT_RUNS_DIR),
        exports_dir=_under_root(exports_dir, ENV_EXPORTS_DIR, DEFAULT_EXPORTS_DIR),
        splits_dir=_under_root(splits_dir, ENV_SPLITS_DIR, DEFAULT_SPLITS_DIR),
    )


def default_paths() -> Paths:
    """Resolve locations from the environment and repo-relative defaults only."""
    return resolve_paths()


#: Mapping of ``argparse`` destination -> :func:`resolve_paths` keyword. The two
#: differ for the audio directory, where ``--audio-dir`` reads better than
#: ``--circor-audio`` and matches the flag the analysis scripts advertise.
_ARGUMENT_TO_KEYWORD: Final[Mapping[str, str]] = {
    "circor_root": "circor_root",
    "circor_meta": "circor_meta",
    "audio_dir": "circor_audio",
    "preds_dir": "preds_dir",
    "results_dir": "results_dir",
    "runs_dir": "runs_dir",
    "exports_dir": "exports_dir",
    "splits_dir": "splits_dir",
}


def add_path_arguments(
    parser: argparse.ArgumentParser,
    *,
    include: tuple[str, ...] = tuple(_ARGUMENT_TO_KEYWORD),
) -> argparse.ArgumentParser:
    """Add the standard path flags to *parser* and return it.

    Pass *include* to expose only the flags a script actually honours, so that
    ``--help`` does not promise an override the script ignores.
    """
    unknown = set(include) - set(_ARGUMENT_TO_KEYWORD)
    if unknown:
        raise ValueError(f"unknown path argument(s): {sorted(unknown)}")
    group = parser.add_argument_group(
        "paths",
        "Each flag overrides the matching environment variable, which in turn "
        "overrides the repo-relative default (see mirasol.paths).",
    )
    help_text: Mapping[str, str] = {
        "circor_root": f"CirCor DigiScope download root (${ENV_CIRCOR_ROOT})",
        "circor_meta": f"CirCor training_data.csv (${ENV_CIRCOR_META})",
        "audio_dir": f"CirCor WAV directory (${ENV_CIRCOR_AUDIO})",
        "preds_dir": f"out-of-fold prediction CSVs (${ENV_PREDS_DIR})",
        "results_dir": f"committed results tree (${ENV_RESULTS_DIR})",
        "runs_dir": f"training run outputs (${ENV_RUNS_DIR})",
        "exports_dir": f"ONNX export outputs (${ENV_EXPORTS_DIR})",
        "splits_dir": f"patient/fold table directory (${ENV_SPLITS_DIR})",
    }
    for name in include:
        group.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=Path,
            default=None,
            metavar="PATH",
            help=help_text[name],
        )
    return parser


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Resolved once at import from the environment and the repo-relative defaults,
# for scripts that only need the default location and never accept a flag (an
# argparse default, a docstring, a hint in an error message). Anything that does
# accept a flag must go through resolve_paths()/paths_from_args() instead, so the
# flag can win; these constants deliberately do not see CLI arguments.

_MODULE_PATHS: Final[Paths] = resolve_paths()

CIRCOR_ROOT: Final[Path] = _MODULE_PATHS.circor_root
CIRCOR_META: Final[Path] = _MODULE_PATHS.circor_meta
CIRCOR_AUDIO: Final[Path] = _MODULE_PATHS.circor_audio
PREDS_DIR: Final[Path] = _MODULE_PATHS.preds_dir
RESULTS_DIR: Final[Path] = _MODULE_PATHS.results_dir
RUNS_DIR: Final[Path] = _MODULE_PATHS.runs_dir
EXPORTS_DIR: Final[Path] = _MODULE_PATHS.exports_dir
SPLITS_DIR: Final[Path] = _MODULE_PATHS.splits_dir
FOLDS_CSV: Final[Path] = _MODULE_PATHS.folds_csv
BENCHMARKS_DIR: Final[Path] = _MODULE_PATHS.benchmarks_dir
LOGS_DIR: Final[Path] = _MODULE_PATHS.logs_dir


def paths_from_args(args: argparse.Namespace) -> Paths:
    """Build a :class:`Paths` from a namespace produced by a parser that was
    passed through :func:`add_path_arguments`.

    Destinations the parser did not define are simply absent, so a script may
    expose any subset of the flags.
    """
    overrides = {
        keyword: getattr(args, dest)
        for dest, keyword in _ARGUMENT_TO_KEYWORD.items()
        if getattr(args, dest, None) is not None
    }
    return resolve_paths(**overrides)
