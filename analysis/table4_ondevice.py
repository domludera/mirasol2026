"""Regenerate Table 4 from the committed on-device benchmark records.

Paper artifact
--------------
Table 4, "On-device benchmark (Orange Pi Zero 2 W, Cortex-A53 x4, 4 GB RAM)",
plus the Sec. 3.4 and Sec. 3.5(1) statements derived from it: the 210 ms / 4.3 MB
EfficientNet-B0 INT8-static row, its 2.2x speed-up over its own FP32 build, the
~1.3-1.4 s transformer latencies, and the dynamic-INT8 slowdown.

Nothing in the original code base regenerated this table -- the numbers were read
out of a JSON by hand -- and the two benchmark records disagree with the paper's
framing in ways a reader deserves to see, so this script prints three blocks:

1. **Table 4 as published**, rebuilt from ``benchmark_2026-02_4arch.json``. The
   selection rule is: the FP32 and INT8-static rows of the four architectures in
   that session, sorted by latency. That is exactly the seven printed rows.
2. **The full six-architecture table** from the later session, which adds
   MobileNetV3-S and ResNet-18. MobileNetV3-S is both faster and smaller than
   the row Table 4 calls "fastest and smallest", and neither of the two later
   architectures appears in the paper at all.
3. **Reconciliations.** The FastViT FP32 size, the per-model dynamic-INT8
   ratios, the FP16 rows, and the peak-RSS caveat. See ``docs/errata.md``.

The two sessions are kept as separate files on purpose; a single merged file
would imply one measurement session five months wide.

Usage
-----
    python -m analysis.table4_ondevice
    python -m analysis.table4_ondevice --benchmarks-dir results/benchmarks --out build
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from analysis._common import (
    ARCH_KEYS,
    print_header,
    print_latex_comment,
    resolve_results_dir,
)

#: Benchmark variant key -> the format label printed in Table 4.
FORMAT_LABELS: Final[Mapping[str, str]] = {
    "model_fp32": "FP32",
    "model_int8_static": "INT8 static",
    "model_int8": "INT8 dynamic",
    "model_fp16": "FP16",
}

#: Formats Table 4 reports. Dynamic INT8 is omitted there (it is slower than FP32
#: on this CPU) and FP16 never ran, so neither is a deployable row.
DEPLOYABLE_FORMATS: Final[tuple[str, ...]] = ("model_fp32", "model_int8_static")

#: Architecture key -> display name, for the four architectures of the February
#: session and the two added later.
DISPLAY_NAMES: Final[Mapping[str, str]] = {key: name for name, key in ARCH_KEYS.items()}

#: Measured size corrections, with the reason each one is needed.
SIZE_CORRECTIONS: Final[Mapping[tuple[str, str], tuple[float, str]]] = {
    ("fastvit", "model_fp32"): (
        12.51,
        "the harness added an orphaned model_fp32.onnx.data (12.44 MiB) to the "
        "12.51 MiB graph; the graph declares no external initializers and runs "
        "standalone, so 24.95 MiB double-counts it",
    ),
}

#: Peak RSS was sampled in a process that had already loaded another model, so
#: these rows report a shared-process high-water mark rather than the model's own.
RSS_SHARED_PROCESS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("tinyvit", "model_fp32"),
        ("tinyvit", "model_int8"),
        ("tinyvit", "model_int8_static"),
        ("htsat", "model_int8"),
    }
)

#: Table 4 prints one integer latency per row. Five of the seven are the rounded
#: mean of the record; two differ by 1 ms because the printed table was rounded by
#: hand in opposite directions. The measurements themselves are unchanged, and
#: this script always prints the rounded mean, which is what the caption promises.
PRINTED_LATENCY_MS: Final[Mapping[tuple[str, str], int]] = {
    ("efficientnet_b0", "model_fp32"): 456,  # mean 456.53 rounds to 457
    ("fastvit", "model_fp32"): 511,  # mean 510.47 rounds to 510
}

DEFAULT_PRIMARY: Final[str] = "benchmark_2026-02_4arch.json"
DEFAULT_EXTRA: Final[str] = "benchmark_2026-07_6arch.json"


@dataclass(frozen=True)
class BenchmarkRow:
    """One (model, format) measurement from a benchmark record."""

    model: str
    variant: str
    size_mib: float | None
    latency_ms: float | None
    rss_peak_mib: float | None
    throughput_ips: float | None
    error: str | None

    @property
    def display(self) -> str:
        return DISPLAY_NAMES.get(self.model, self.model)

    @property
    def format_label(self) -> str:
        return FORMAT_LABELS.get(self.variant, self.variant)

    @property
    def ran(self) -> bool:
        return self.error is None and self.latency_ms is not None


@dataclass(frozen=True)
class BenchmarkSession:
    """A whole benchmark record: device metadata plus its rows."""

    path: Path
    device: str
    cpu: str
    timestamp: str
    runs: int
    warmup: int
    rows: tuple[BenchmarkRow, ...]

    def find(self, model: str, variant: str) -> BenchmarkRow | None:
        for row in self.rows:
            if row.model == model and row.variant == variant:
                return row
        return None


def load_session(path: Path) -> BenchmarkSession:
    """Read one benchmark JSON into an immutable session record."""
    payload = json.loads(path.read_text())
    for key in ("results", "device", "runs", "warmup"):
        if key not in payload:
            raise SystemExit(
                f"error: '{path.name}' is missing the '{key}' field; expected a record "
                "written by deployment/edge/benchmark.py."
            )
    rows = tuple(
        BenchmarkRow(
            model=str(entry["model"]),
            variant=str(entry["variant"]),
            size_mib=entry.get("model_size_mb"),
            latency_ms=entry.get("latency_avg_ms"),
            rss_peak_mib=entry.get("rss_peak_mb"),
            throughput_ips=entry.get("throughput_ips"),
            error=entry.get("error"),
        )
        for entry in payload["results"]
    )
    return BenchmarkSession(
        path=path,
        device=str(payload["device"]),
        cpu=str(payload.get("cpu", "unknown")),
        timestamp=str(payload.get("timestamp", "unknown")),
        runs=int(payload["runs"]),
        warmup=int(payload["warmup"]),
        rows=rows,
    )


def deployable_rows(session: BenchmarkSession) -> list[BenchmarkRow]:
    """FP32 and INT8-static rows that actually ran, sorted by latency."""
    rows = [
        row
        for row in session.rows
        if row.variant in DEPLOYABLE_FORMATS and row.ran
    ]
    return sorted(rows, key=lambda row: row.latency_ms or float("inf"))


def _size(row: BenchmarkRow, *, corrected: bool) -> float | None:
    if corrected and (row.model, row.variant) in SIZE_CORRECTIONS:
        return SIZE_CORRECTIONS[(row.model, row.variant)][0]
    return row.size_mib


def latex_table(rows: Sequence[BenchmarkRow], *, corrected: bool = False) -> str:
    """The Table 4 tabular body, bolding the best cell in each numeric column."""
    if not rows:
        raise SystemExit("error: no deployable benchmark rows to tabulate.")
    best_latency = min(row.latency_ms for row in rows if row.latency_ms is not None)
    sizes = [s for s in (_size(row, corrected=corrected) for row in rows) if s is not None]
    best_size = min(sizes) if sizes else None
    rss = [row.rss_peak_mib for row in rows if row.rss_peak_mib is not None]
    best_rss = min(rss) if rss else None
    throughputs = [row.throughput_ips for row in rows if row.throughput_ips is not None]
    best_throughput = max(throughputs) if throughputs else None

    def bold(text: str, is_best: bool) -> str:
        return rf"\textbf{{{text}}}" if is_best else text

    lines = []
    for row in rows:
        latency = row.latency_ms or 0.0
        latency_text = f"{round(latency):,}".replace(",", "{,}")
        size = _size(row, corrected=corrected)
        size_text = "---" if size is None else f"{size:.1f}"
        rss_text = "---" if row.rss_peak_mib is None else f"{round(row.rss_peak_mib)}"
        throughput = row.throughput_ips
        throughput_text = "---" if throughput is None else f"{throughput:.2f} inf/s"
        lines.append(
            " & ".join(
                [
                    row.display,
                    row.format_label,
                    bold(latency_text, latency == best_latency),
                    bold(size_text, size is not None and size == best_size),
                    bold(rss_text, row.rss_peak_mib == best_rss),
                    bold(throughput_text, throughput == best_throughput),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def print_session_header(session: BenchmarkSession, title: str) -> None:
    """State the provenance of every number that follows."""
    print_header(title)
    print(f"  record   {session.path.name}")
    print(f"  device   {session.device} ({session.cpu})")
    print(f"  measured {session.timestamp}")
    print(f"  protocol mean of {session.runs} runs after {session.warmup} warmups")


def print_ascii_table(rows: Sequence[BenchmarkRow], *, corrected: bool = False) -> None:
    """Human-readable form of the same rows, ASCII only."""
    header = (
        f"  {'Model':<16}{'Format':<13}{'Latency (ms)':>13}{'Size (MiB)':>12}"
        f"{'Peak RSS (MiB)':>16}{'Throughput':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        size = _size(row, corrected=corrected)
        note = " *" if (row.model, row.variant) in RSS_SHARED_PROCESS else ""
        print(
            f"  {row.display:<16}{row.format_label:<13}"
            f"{round(row.latency_ms or 0):>13,}"
            f"{'---' if size is None else f'{size:.2f}':>12}"
            f"{'---' if row.rss_peak_mib is None else f'{row.rss_peak_mib:.1f}':>16}"
            f"{'---' if row.throughput_ips is None else f'{row.throughput_ips:.3f}':>13}"
            f"{note}"
        )
    if any((row.model, row.variant) in RSS_SHARED_PROCESS for row in rows):
        print(
            "  * peak RSS is a shared-process high-water mark: the harness kept one\n"
            "    process alive across models, so these rows inherit the peak of a\n"
            "    previously loaded graph. deployment/edge/benchmark.py now reads VmHWM\n"
            "    and runs one model per subprocess."
        )


def print_latency_rounding_notes(rows: Sequence[BenchmarkRow]) -> None:
    """Name the cells where the printed integer differs from the rounded mean."""
    divergent = [
        (row, PRINTED_LATENCY_MS[(row.model, row.variant)])
        for row in rows
        if (row.model, row.variant) in PRINTED_LATENCY_MS
        and row.latency_ms is not None
        and round(row.latency_ms) != PRINTED_LATENCY_MS[(row.model, row.variant)]
    ]
    if not divergent:
        return
    print(
        f"% note: {len(divergent)} printed latency integer(s) differ by 1 ms from the\n"
        "% rounded mean of the record, an artifact of rounding the table by hand:"
    )
    for row, printed in divergent:
        print(
            f"%   {row.display} {row.format_label}: printed {printed}, "
            f"mean {row.latency_ms:.2f} -> {round(row.latency_ms)}"
        )


def print_dynamic_int8_ratios(session: BenchmarkSession) -> None:
    """Per-model dynamic-INT8 / FP32 latency ratios measured on the device."""
    print()
    print_header("Dynamic INT8 versus FP32 on the device")
    print(
        "  The Table 4 footnote's '2-7x slower' range is the desktop x86\n"
        "  measurement. On the Cortex-A53 the per-model ratios are:"
    )
    for model in DISPLAY_NAMES:
        fp32 = session.find(model, "model_fp32")
        dynamic = session.find(model, "model_int8")
        if fp32 is None or dynamic is None or not (fp32.ran and dynamic.ran):
            continue
        ratio = (dynamic.latency_ms or 0.0) / (fp32.latency_ms or 1.0)
        verdict = "slower" if ratio > 1.0 else "FASTER than FP32"
        print(
            f"  {DISPLAY_NAMES[model]:<16} {dynamic.latency_ms:8.1f} ms vs "
            f"{fp32.latency_ms:8.1f} ms  ->  {ratio:.2f}x {verdict}"
        )


def print_fp16_status(session: BenchmarkSession) -> None:
    """FP16 rows are harness dtype failures, not hardware evidence."""
    failures = [
        row for row in session.rows if row.variant == "model_fp16" and row.error
    ]
    if not failures:
        return
    print()
    print_header("FP16 rows")
    print(
        "  Every FP16 entry in this record is a harness failure, not a hardware\n"
        "  limitation: the benchmark fed float32 into a float16 graph. The Sec. 3.5(2)\n"
        "  FP16 agreement claims come from the desktop validation logs in\n"
        "  results/logs/static_fp16.log, not from this device record."
    )
    for row in failures:
        print(f"  {row.display:<16} {(row.error or '')[:88]}")


def print_size_corrections(rows: Sequence[BenchmarkRow]) -> None:
    """Documented size corrections, with the reason for each."""
    applicable = [r for r in rows if (r.model, r.variant) in SIZE_CORRECTIONS]
    if not applicable:
        return
    print()
    print_header("Model-size reconciliation")
    for row in applicable:
        corrected, reason = SIZE_CORRECTIONS[(row.model, row.variant)]
        print(
            f"  {row.display} {row.format_label}: recorded "
            f"{row.size_mib:.2f} MiB, actual {corrected:.2f} MiB"
        )
        print(f"    {reason}")


def print_missing_architectures(
    primary: BenchmarkSession, extra: BenchmarkSession
) -> None:
    """Architectures measured later that Table 4 does not contain."""
    measured_first = {row.model for row in primary.rows}
    added = [
        model for model in DISPLAY_NAMES
        if model not in measured_first and any(r.model == model for r in extra.rows)
    ]
    if not added:
        return
    print()
    print_header("Architectures absent from Table 4")
    print(
        "  Table 4 predates these measurements, so it does not contain them even\n"
        "  though the paper benchmarks six architectures elsewhere:"
    )
    reference = primary.find("efficientnet_b0", "model_int8_static")
    for model in added:
        for variant in DEPLOYABLE_FORMATS:
            row = extra.find(model, variant)
            if row is None or not row.ran:
                continue
            comparison = ""
            if reference is not None and reference.ran:
                faster = (row.latency_ms or 0.0) < (reference.latency_ms or 0.0)
                smaller = (row.size_mib or 0.0) < (reference.size_mib or 0.0)
                if faster and smaller:
                    comparison = (
                        "  <- faster AND smaller than the EfficientNet-B0 INT8-static "
                        "row Table 4 calls fastest and smallest"
                    )
                elif faster:
                    comparison = "  <- faster than that EfficientNet-B0 INT8-static row"
            print(
                f"  {row.display:<16}{row.format_label:<13}"
                f"{row.latency_ms:8.1f} ms  {row.size_mib:6.2f} MiB{comparison}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.table4_ondevice",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory of benchmark records (default: results/benchmarks)",
    )
    parser.add_argument(
        "--primary",
        default=DEFAULT_PRIMARY,
        metavar="FILE",
        help="the session Table 4 was built from",
    )
    parser.add_argument(
        "--extra",
        default=DEFAULT_EXTRA,
        metavar="FILE",
        help="the later six-architecture session",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write table4.tex and table4_corrected.tex to DIR",
    )
    return parser


def _require_record(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        available = sorted(p.name for p in directory.glob("*.json"))
        raise SystemExit(
            f"error: benchmark record '{name}' not found in "
            f"'{directory.name}'. Available: {available or 'none'}."
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = resolve_results_dir(args.benchmarks_dir, "benchmarks")

    primary = load_session(_require_record(directory, args.primary))
    extra = load_session(_require_record(directory, args.extra))

    published = deployable_rows(primary)
    print_session_header(primary, "Table 4 as published: deployable formats, four architectures")
    print_ascii_table(published)
    print_latex_comment("TABLE 4 -- as published")
    print(latex_table(published))
    print_latency_rounding_notes(published)

    print_size_corrections(published)
    print_latex_comment("TABLE 4 -- with the measured FastViT FP32 size")
    print(latex_table(published, corrected=True))

    print()
    print_session_header(extra, "Six-architecture session, deployable formats")
    print_ascii_table(deployable_rows(extra))

    print_dynamic_int8_ratios(primary)
    print_fp16_status(primary)
    print_missing_architectures(primary, extra)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "table4.tex").write_text(latex_table(published) + "\n")
        (args.out / "table4_corrected.tex").write_text(
            latex_table(published, corrected=True) + "\n"
        )
        (args.out / "table4_six_arch.tex").write_text(
            latex_table(deployable_rows(extra)) + "\n"
        )
        print(f"\nwrote three table bodies to {args.out.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
