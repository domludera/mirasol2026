"""Regenerate Figure 1 and Figure S1.

Paper artifacts
---------------
``preprocessing_pipeline.png`` -- Figure 1 of the paper: a four-panel
illustration ending in the log-mel spectrogram actually fed to the network, with
the tensor shape annotated at each stage.

``clinical_endpoint.png`` -- Figure S1 of the supplement: (a) the sensitivity /
referral-burden trade-off against the examining cardiologist's diagnosis with each
model's deployed operating point and the expert-annotation ceiling marked, and
(b) murmur-detection sensitivity stratified by auscultatory grade with Wilson 95%
intervals.

Colours are the Okabe-Ito colour-vision-deficiency-safe set, assigned to models
in fixed parameter-count order and backed by distinct markers and dash patterns,
so identity never rests on hue alone.

Requirements differ per figure
------------------------------
Figure S1 needs only the committed prediction dumps plus CirCor's 114 KB
``training_data.csv``. Figure 1 additionally needs one recording from the ~560 MB
audio download, so the two are separately selectable with ``--figures``:
``--figures clinical`` renders Figure S1 without any audio at all.

Signal processing
-----------------
The bandpass, z-score and log-mel front end are imported from
:mod:`mirasol.signal.mel`, the single definition of the paper's model input. That
bandpass is the causal ``sosfilt`` the training pipeline uses. An earlier
standalone copy of this figure applied a zero-phase ``sosfiltfilt``, so panels
(b)-(d) of the published Figure 1 show a filter with a different phase response
from the one the models were trained with; the magnitude response, and hence the
visual content of the figure, is unchanged.

Usage
-----
    python -m analysis.figures --figures clinical
    python -m analysis.figures --audio-dir path/to/training_data --out fig
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

from mirasol.signal.mel import bandpass, tile_to_length, waveform_to_mel_spectrogram, zscore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be set first)

from analysis._common import (  # noqa: E402
    HIGH_GRADE_SYSTOLIC,
    add_audio_dir_argument,
    add_circor_meta_argument,
    add_preds_dir_argument,
    resolve_audio_dir,
    resolve_circor_meta,
    resolve_preds_dir,
)
from analysis.clinical_endpoint import (  # noqa: E402
    PRESENT,
    endpoint_mask,
    load_all_models,
    load_reference,
    referral_mask,
    screening_metrics,
    sweep_curve,
    wilson_ci,
)

# Preprocessing constants -- identical to mirasol.config.defaults.
SAMPLE_RATE: Final[int] = 4000
BAND: Final[tuple[float, float]] = (25.0, 400.0)
CLIP_SECONDS: Final[int] = 30
HOP: Final[int] = 120
MEL_BINS: Final[int] = 64
FMIN: Final[int] = 25
FMAX: Final[int] = 2000

#: Display dynamic range of panel (d), in dB below the panel maximum.
DISPLAY_TOP_DB: Final[float] = 60.0

#: Default example: a grade III/VI holosystolic harsh murmur at the tricuspid
#: position (patient 9979, expert outcome Abnormal).
DEFAULT_RECORDING: Final[str] = "9979_TV.wav"

#: Written as an escape so this source file stays pure ASCII while the rendered
#: figure keeps the typographic dash of the published version.
EN_DASH: Final[str] = "\u2013"

# Okabe-Ito, validated for colour-vision-deficiency separation in this order.
PALETTE: Final[tuple[str, ...]] = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
)
MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D", "v", "P")
DASHES: Final[tuple[object, ...]] = (
    (None, None), (5, 1.6), (1.4, 1.4), (6, 1.6, 1.4, 1.6), (3, 1.4), (8, 2),
)

INK: Final[str] = "#1a1a1a"
MUTED: Final[str] = "#6b6b6b"
GRID: Final[str] = "#d9d9d9"

RC_PARAMS: Final[Mapping[str, object]] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "savefig.dpi": 400,
    "figure.dpi": 400,
}


def _style(ax: plt.Axes) -> None:
    """Recessive frame and grid so the data carries the ink."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


# --------------------------------------------------------------------------
# Figure 1: preprocessing pipeline
# --------------------------------------------------------------------------
def cleanest_window(signal: np.ndarray, seconds: float, limit: int) -> int:
    """Start index of the window least dominated by transient spikes.

    CirCor recordings contain sporadic contact artefacts an order of magnitude
    above the heart sounds; plotting one of those hides the signal the figure is
    meant to show. Windows are scored by peak-to-r.m.s. ratio, lower being calmer.

    ``limit`` bounds the search so the chosen window lies inside the first tile of
    the model input. Panel (d) indexes the spectrogram of the tiled 30 s clip by
    ``start // HOP``, so a start past the first tile would show a mel window that
    does not correspond to the waveform panels beside it.
    """
    width = int(seconds * SAMPLE_RATE)
    highest = min(limit, signal.size) - width
    if highest <= 0:
        return 0
    step = max(1, SAMPLE_RATE // 4)

    def score(start: int) -> float:
        window = signal[start:start + width]
        rms = float(np.sqrt(np.mean(window ** 2)))
        return float(np.max(np.abs(window))) / rms if rms > 0 else float("inf")

    return min(range(0, highest, step), key=score)


def model_input(clip: np.ndarray) -> np.ndarray:
    """The 30 s clip concatenated with itself, as the data loader does.

    With hop 120 this is what yields the 2001 frames the paper reports rather
    than the 1001 frames a single 30 s clip would give.
    """
    return np.concatenate((clip, clip))


def display_mel(waveform: np.ndarray) -> np.ndarray:
    """Log-mel input of shape ``(mel_bins, frames)``, referenced to its maximum.

    The values are the training log-mel exactly; only the reference level is
    shifted (and the range floored at ``DISPLAY_TOP_DB``) so that the colour scale
    reads as dB below the panel peak.
    """
    log_mel = waveform_to_mel_spectrogram(waveform)[0].T
    shifted = log_mel - log_mel.max()
    return np.maximum(shifted, -DISPLAY_TOP_DB)


def figure_preprocessing(
    audio_dir: Path, out_dir: Path, recording: str = DEFAULT_RECORDING,
    seconds: float = 3.0,
) -> Path:
    """Figure 1: the four-stage preprocessing illustration."""
    path = audio_dir / recording
    if not path.is_file():
        raise SystemExit(
            f"error: recording '{recording}' not found in '{audio_dir.name}'. "
            "Figure 1 needs one CirCor recording; Figure S1 does not."
        )
    raw, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        import librosa

        raw = librosa.resample(raw, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    filtered = bandpass(raw, sr=SAMPLE_RATE, low=BAND[0], high=BAND[1])
    normalised = zscore(filtered)
    clip = tile_to_length(normalised, CLIP_SECONDS * SAMPLE_RATE)
    mel = display_mel(model_input(clip))

    show = int(seconds * SAMPLE_RATE)
    start = cleanest_window(normalised, seconds, limit=CLIP_SECONDS * SAMPLE_RATE)
    time = np.arange(show) / SAMPLE_RATE

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.55))
        samples = f"$1\\times{SAMPLE_RATE * CLIP_SECONDS:,}$".replace(",", "{,}")
        panels = (
            (raw, "(a) Raw PCG\n4 kHz", samples),
            (filtered, f"(b) Bandpass\n{BAND[0]:.0f}{EN_DASH}{BAND[1]:.0f} Hz", samples),
            (normalised, "(c) $z$-score\nnormalisation", samples),
        )
        # 4 axes but 3 waveform panels: the fourth axis holds the mel spectrogram,
        # drawn below, so this pairing is deliberately unequal.
        for ax, (signal, title, shape) in zip(axes, panels, strict=False):
            ax.plot(time, signal[start:start + show], color=PALETTE[0], linewidth=0.35)
            ax.set_title(title, pad=3)
            ax.set_xlabel("Time (s)", labelpad=1.5)
            ax.set_xlim(0, seconds)
            ax.text(0.98, 0.05, shape, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.2, color=MUTED)
            _style(ax)
        axes[0].set_ylabel("Amplitude", labelpad=1.5)

        ax = axes[3]
        first_frame = start // HOP
        frames = int(seconds * SAMPLE_RATE / HOP)
        image = ax.imshow(
            mel[:, first_frame:first_frame + frames], aspect="auto", origin="lower",
            cmap="magma", extent=(0.0, seconds, 0, MEL_BINS),
        )
        ax.set_title(
            f"(d) Log-mel input\n{MEL_BINS} mels, {FMIN}{EN_DASH}{FMAX} Hz", pad=3
        )
        ax.set_xlabel("Time (s)", labelpad=1.5)
        ax.set_ylabel("Mel bin", labelpad=1.5)
        ax.text(0.98, 0.04, f"$1\\times{mel.shape[1]}\\times{MEL_BINS}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
                color="white")
        colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label("dB", labelpad=1.5, fontsize=7)
        colorbar.ax.tick_params(labelsize=6, length=2, width=0.5)
        colorbar.outline.set_linewidth(0.5)

        fig.tight_layout(pad=0.35, w_pad=1.1)
        out = out_dir / "preprocessing_pipeline.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out.name}  (mel shape 1 x {mel.shape[1]} x {MEL_BINS})")
    return out


# --------------------------------------------------------------------------
# Figure S1: clinical endpoint
# --------------------------------------------------------------------------
#: Strata of panel (b): three ordinal systolic grades, then diastolic as its own
#: category. Keys are the stratum keys of :mod:`analysis.clinical_endpoint`.
GRADE_STRATA: Final[tuple[tuple[str, str], ...]] = (
    ("I/VI", "grade\nI/VI"),
    ("II/VI", "grade\nII/VI"),
    (">=III/VI", "grade\n$\\geq$III/VI"),
    ("diastolic", "any\ndiastolic"),
)

#: Systolic grade is ordinal, so those three points are joined; "any diastolic"
#: is a separate category, drawn detached past a divider.
N_ORDINAL: Final[int] = 3


def _grade_mask(preds: pd.DataFrame, key: str) -> np.ndarray:
    if key == "diastolic":
        return preds["has_diastolic"].to_numpy(dtype=bool)
    if key == ">=III/VI":
        return preds["systolic_grade"].isin(HIGH_GRADE_SYSTOLIC).to_numpy(dtype=bool)
    return (preds["systolic_grade"] == key).to_numpy(dtype=bool)


def _panel_tradeoff(ax: plt.Axes, frames: Mapping[str, pd.DataFrame]) -> None:
    """Panel (a): sensitivity against referral burden for endpoint E1."""
    for index, (name, preds) in enumerate(frames.items()):
        curve = sweep_curve(preds, "abnormal_cardiac_status")
        burden = np.array([row["referral_rate"] for row in curve])
        sensitivity = np.array([row["sensitivity"] for row in curve])
        order = np.argsort(burden)
        ax.plot(burden[order], sensitivity[order], color=PALETTE[index],
                linewidth=1.0, dashes=DASHES[index], label=name, zorder=3)
        deployed = screening_metrics(
            referral_mask(preds, "model_present"),
            endpoint_mask(preds, "abnormal_cardiac_status"),
        )
        ax.plot(deployed.referral_rate, deployed.sensitivity, MARKERS[index],
                color=PALETTE[index], markersize=4.2, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)

    any_model = next(iter(frames.values()))
    expert = screening_metrics(
        referral_mask(any_model, "expert_annotation"),
        endpoint_mask(any_model, "abnormal_cardiac_status"),
    )
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.7, dashes=(2, 2), zorder=1)
    ax.annotate("refer at random", xy=(0.62, 0.62), xytext=(0.60, 0.50),
                fontsize=6.3, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5))
    ax.plot(expert.referral_rate, expert.sensitivity, "*", color=INK, markersize=9,
            markeredgecolor="white", markeredgewidth=0.6, zorder=6)
    ax.annotate(
        "expert murmur\nannotation (ceiling)",
        xy=(expert.referral_rate, expert.sensitivity),
        xytext=(expert.referral_rate + 0.10, expert.sensitivity - 0.17),
        fontsize=6.3, color=INK, ha="left",
        arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.5),
    )
    ax.set_xlabel("Referral burden (fraction of screened patients)", labelpad=2)
    ax.set_ylabel("Sensitivity for abnormal cardiac status", labelpad=2)
    ax.set_title("(a) Referral trade-off, expert-cardiologist endpoint", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4, borderpad=0.2,
              labelspacing=0.28)


def _panel_severity(ax: plt.Axes, frames: Mapping[str, pd.DataFrame]) -> None:
    """Panel (b): detection sensitivity by auscultatory severity."""
    x = np.arange(len(GRADE_STRATA), dtype=float)
    counts: list[int] = []
    for index, (name, preds) in enumerate(frames.items()):
        sensitivity, lower, upper = [], [], []
        for key, _ in GRADE_STRATA:
            subset = preds[_grade_mask(preds, key)]
            n = len(subset)
            detected = int((subset["murmur_pred"] == PRESENT).sum())
            value = detected / n if n else np.nan
            interval = wilson_ci(detected, n)
            sensitivity.append(value)
            lower.append(value - interval[0])
            upper.append(interval[1] - value)
            if index == 0:
                counts.append(n)
        offset = (index - (len(frames) - 1) / 2) * 0.11
        shared = dict(color=PALETTE[index], marker=MARKERS[index], markersize=3.8,
                      markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.5,
                      capsize=1.4, capthick=0.5, zorder=3)
        ax.errorbar(
            x[:N_ORDINAL] + offset, sensitivity[:N_ORDINAL],
            yerr=[lower[:N_ORDINAL], upper[:N_ORDINAL]],
            linewidth=0.9, dashes=DASHES[index], label=name, **shared,
        )
        ax.errorbar(
            x[N_ORDINAL:] + offset, sensitivity[N_ORDINAL:],
            yerr=[lower[N_ORDINAL:], upper[N_ORDINAL:]], linestyle="none", **shared,
        )
    ax.axvline(N_ORDINAL - 0.5, color=GRID, linewidth=0.6, dashes=(2, 2))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n$n={n}$" for (_, label), n in zip(GRADE_STRATA, counts, strict=True)]
    )
    ax.set_ylabel("Murmur-detection sensitivity", labelpad=2)
    ax.set_title("(b) Sensitivity by auscultatory severity", pad=4)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(GRADE_STRATA) - 0.5)
    ax.axhline(1.0, color=GRID, linewidth=0.4)
    _style(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4, borderpad=0.2,
              labelspacing=0.28, ncol=2, columnspacing=0.9)


def figure_clinical_endpoint(
    frames: Mapping[str, pd.DataFrame], out_dir: Path
) -> Path:
    """Figure S1: the two-panel clinical-utility figure."""
    with plt.rc_context(RC_PARAMS):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.6, 1.85))
        _panel_tradeoff(ax_left, frames)
        _panel_severity(ax_right, frames)
        fig.tight_layout(pad=0.4, w_pad=1.4)
        out = out_dir / "clinical_endpoint.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out.name}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.figures",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preds_dir_argument(parser)
    add_circor_meta_argument(parser)
    add_audio_dir_argument(parser)
    parser.add_argument(
        "--figures",
        choices=("preprocessing", "clinical", "all"),
        default="all",
        help="which figures to render; 'clinical' needs no audio download",
    )
    parser.add_argument(
        "--recording",
        default=DEFAULT_RECORDING,
        metavar="WAV",
        help="recording used for the Figure 1 panels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fig"),
        metavar="DIR",
        help="output directory for the PNGs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figures in ("preprocessing", "all"):
        figure_preprocessing(
            resolve_audio_dir(args.audio_dir), out_dir, recording=args.recording
        )
    if args.figures in ("clinical", "all"):
        reference = load_reference(resolve_circor_meta(args.circor_meta))
        frames = load_all_models(resolve_preds_dir(args.preds_dir), reference)
        figure_clinical_endpoint(frames, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
