"""Machine-verify every reference in the manuscript against Crossref and arXiv.

What this checks
----------------
The bibliography is checked entry by entry rather than trusted: each DOI must
resolve at Crossref, each arXiv identifier must resolve at the arXiv API, and the
resolved metadata must agree with what the manuscript prints -- title, first
author, year, volume, pages and container title. Anything with no machine-checkable
identifier is listed explicitly rather than passed over in silence, so the set of
entries a human still has to confirm is always visible.

Serves the reference list of "A Low-Cost AI Stethoscope for Resource-Constrained
Heart Murmur Screening" (MIRASOL @ MICCAI 2026): 35 entries, of which 34 carry a
DOI or arXiv identifier and one is a WHO data portal cited by URL.

Design notes
------------
* Titles are matched against every ``". "``-delimited segmentation of the printed
  entry, because LNCS entries legitimately contain sentence periods inside a title
  ("The George B. Moody PhysioNet Challenge 2022"). A title that does not exist
  matches no segmentation, so this loosening cannot hide a wrong one.
* The printed first-author family name is matched against *all* resolved family
  names, with token containment, because Crossref records compound families
  ("Anasosalu Vasu") and sometimes a different author order.
* A one-year difference between the printed year and the Crossref year is normal
  for online-first publication and is reported as information, not as a problem.
* Transport failures are separated from "this identifier does not exist": each
  lookup is retried with backoff, and the two outcomes get different exit codes,
  so a flaky network never looks like a bad reference.

Exit codes
----------
0  every identifier resolved and every field agreed
1  at least one identifier does not exist, or its metadata disagrees
2  at least one lookup could not be completed (network or rate limiting)

Usage
-----
    python -m analysis.verify_references path/to/MIRASOL_023.tex

The manuscript source is not redistributed in this repository, so the path is
normally given explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Final

from mirasol.paths import REPO_ROOT

#: Default location of the manuscript source, relative to the repository root.
DEFAULT_TEX: Final[Path] = REPO_ROOT / "camera_ready" / "MIRASOL_023.tex"

#: Crossref asks for a contact address in the User-Agent to use the polite pool.
#: This is the corresponding author's published address.
CONTACT: Final[str] = "dominik.ludera@mail.concordia.ca"
USER_AGENT: Final[str] = f"MIRASOL-RefCheck/2.0 (mailto:{CONTACT})"

CROSSREF: Final[str] = "https://api.crossref.org/works/"
ARXIV: Final[str] = "http://export.arxiv.org/api/query?id_list="

TITLE_MATCH_FLOOR: Final[float] = 0.82
FAMILY_MATCH_FLOOR: Final[float] = 0.85

ATTEMPTS: Final[int] = 3
BACKOFF_SECONDS: Final[float] = 1.5
POLITE_DELAY_SECONDS: Final[float] = 0.12
TIMEOUT_SECONDS: Final[int] = 30


class LookupFailed(Exception):
    """A lookup could not be completed; the reference itself is unjudged."""


class NotFound(Exception):
    """The identifier resolved to nothing: the reference does not exist."""


@dataclass(frozen=True)
class Entry:
    """One ``\\bibitem`` of the manuscript bibliography."""

    key: str
    text: str

    @property
    def doi(self) -> str | None:
        match = re.search(r"\\doi\{(.+?)\}", self.text)
        return match.group(1) if match else None

    @property
    def arxiv(self) -> str | None:
        match = re.search(r"arXiv:([0-9]{4}\.[0-9]{4,5})", self.text)
        return match.group(1) if match else None

    @property
    def url(self) -> str | None:
        match = re.search(r"\\url\{(.+?)\}", self.text)
        return match.group(1) if match else None

    @property
    def printed_year(self) -> int | None:
        years = re.findall(r"\((\d{4})\)", self.text)
        return int(years[-1]) if years else None

    @property
    def first_author_family(self) -> str:
        """``"Family, I.N., Other, A.: Title..."`` -> ``"Family"``."""
        head = self.text.split(":")[0]
        return head.split(",")[0].strip().strip("{}")

    @property
    def title_candidates(self) -> tuple[str, ...]:
        """Every plausible printed title, as ``". "``-delimited prefixes.

        The body after the first ``": "`` begins with the title and continues into
        the venue. A title may itself contain ``". "``, so instead of guessing one
        cut point every cumulative prefix ending at a ``". "`` boundary is offered
        and the best match against the resolved title wins.
        """
        flat = " ".join(self.text.split())
        body = flat.split(": ", 1)[1] if ": " in flat else flat
        body = re.split(r"\\doi\{|\\url\{", body)[0]
        body = body.split(". In: ")[0]
        pieces = body.split(". ")
        candidates: list[str] = []
        for index in range(1, len(pieces) + 1):
            candidate = _clean(". ".join(pieces[:index]))
            if candidate:
                candidates.append(candidate)
        return tuple(candidates)


def _clean(text: str) -> str:
    """Lowercase alphanumeric form used for every fuzzy comparison."""
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def similar(left: str, right: str) -> float:
    """Similarity of two strings after cleaning."""
    return SequenceMatcher(None, _clean(left), _clean(right)).ratio()


def best_title_match(candidates: Sequence[str], resolved: str) -> float:
    """Best similarity over the candidate titles, allowing for truncation.

    Crossref sometimes stores a shortened title (only the part before a colon), so
    a candidate is also compared against its own leading substring of the resolved
    title's length.
    """
    target = _clean(resolved)
    if not target or not candidates:
        return 0.0
    best = 0.0
    for candidate in candidates:
        best = max(best, similar(candidate, target))
        if len(candidate) > len(target):
            best = max(best, similar(candidate[: len(target)], target))
    return best


def family_matches(printed: str, families: Sequence[str]) -> bool:
    """Whether the printed family name appears among the resolved authors.

    Compound families ("Anasosalu Vasu") and reordered author lists both occur in
    Crossref records, so every resolved family is considered and a token-level
    containment counts as a match.
    """
    needle = _clean(printed)
    if not needle:
        return True
    needle_tokens = set(needle.split())
    for family in families:
        candidate = _clean(family)
        if not candidate:
            continue
        if similar(needle, candidate) > FAMILY_MATCH_FLOOR:
            return True
        if needle_tokens and needle_tokens <= set(candidate.split()):
            return True
    return False


def fetch(url: str) -> bytes:
    """Fetch a URL, retrying transport failures and distinguishing a 404."""
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                raise NotFound(f"HTTP {error.code}") from error
            last = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
        if attempt + 1 < ATTEMPTS:
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise LookupFailed(f"{type(last).__name__}: {last}")


def parse_entries(path: Path) -> list[Entry]:
    """Split the manuscript's ``thebibliography`` environment into entries."""
    body = path.read_text()
    if r"\begin{thebibliography}" not in body:
        raise SystemExit(
            f"error: '{path.name}' contains no thebibliography environment; point "
            "this script at the manuscript .tex source."
        )
    body = body.split(r"\begin{thebibliography}")[1].split(r"\end{thebibliography}")[0]
    chunks = re.split(r"\\bibitem\{(.+?)\}", body)[1:]
    return [
        Entry(key=chunks[index], text=" ".join(chunks[index + 1].split()))
        for index in range(0, len(chunks), 2)
    ]


def check_crossref(entry: Entry) -> tuple[list[str], list[str]]:
    """Compare an entry against its Crossref record.

    Returns ``(problems, information)``. ``LookupFailed`` and ``NotFound``
    propagate so the caller can tell a flaky network from a bad DOI.
    """
    doi = entry.doi
    assert doi is not None  # only called when a DOI is present
    payload = json.loads(fetch(CROSSREF + urllib.parse.quote(doi, safe="/")))
    message = payload["message"]
    problems: list[str] = []
    information: list[str] = []

    resolved_title = (message.get("title") or [""])[0]
    ratio = best_title_match(entry.title_candidates, resolved_title)
    if ratio < TITLE_MATCH_FLOOR:
        problems.append(
            f"title {ratio:.2f}: resolved '{resolved_title[:70]}' matches no "
            f"segmentation of the printed entry"
        )

    families = [author.get("family", "") for author in message.get("author", [])]
    if families and not family_matches(entry.first_author_family, families):
        problems.append(
            f"first author: printed '{entry.first_author_family}' is not among the "
            f"resolved authors {families[:4]}"
        )

    parts = (message.get("issued", {}).get("date-parts") or [[None]])[0]
    resolved_year = parts[0] if parts else None
    printed_year = entry.printed_year
    if printed_year and resolved_year and printed_year != resolved_year:
        if abs(printed_year - resolved_year) > 1:
            problems.append(
                f"year: printed {printed_year} vs resolved {resolved_year}"
            )
        else:
            information.append(
                f"year off by one, normal for online-first: printed {printed_year}, "
                f"Crossref {resolved_year}"
            )

    printed_digits = re.sub(r"[^0-9]", "", entry.text)
    for field, label in (("volume", "volume"), ("page", "pages")):
        value = message.get(field)
        if not value or str(value) in entry.text.replace("--", "-"):
            continue
        digits = re.sub(r"[^0-9]", "", str(value))
        if digits and digits not in printed_digits:
            problems.append(f"{label} '{value}' not found in the printed entry")

    container = (message.get("container-title") or [""])[0]
    if container:
        tokens = [t for t in _clean(container).split() if len(t) > 3]
        printed = _clean(entry.text)
        if tokens and not any(token in printed for token in tokens):
            problems.append(
                f"venue '{container[:45]}' not recognisable in the entry"
            )
    return problems, information


def check_arxiv(entry: Entry) -> tuple[list[str], list[str]]:
    """Compare an entry against its arXiv record."""
    root = ET.fromstring(fetch(ARXIV + (entry.arxiv or "")))
    namespace = {"a": "http://www.w3.org/2005/Atom"}
    titles = root.findall(".//a:entry/a:title", namespace)
    if not titles:
        raise NotFound("arXiv identifier returned no entry")
    resolved = titles[0].text or ""
    ratio = best_title_match(entry.title_candidates, resolved)
    problems = (
        [
            f"title {ratio:.2f}: arXiv '{_clean(resolved)[:70]}' matches no "
            "segmentation of the printed entry"
        ]
        if ratio < TITLE_MATCH_FLOOR
        else []
    )
    return problems, []


@dataclass(frozen=True)
class Outcome:
    """Result of checking one entry."""

    entry: Entry
    identifier: str
    status: str
    problems: tuple[str, ...]
    information: tuple[str, ...]


def check(entry: Entry) -> Outcome:
    """Check one entry, converting lookup exceptions into an outcome."""
    if entry.doi:
        identifier, checker = f"doi:{entry.doi}", check_crossref
    elif entry.arxiv:
        identifier, checker = f"arXiv:{entry.arxiv}", check_arxiv
    else:
        return Outcome(entry, "(no DOI/arXiv id)", "NEEDS MANUAL CHECK", (), ())
    try:
        problems, information = checker(entry)
    except NotFound as error:
        return Outcome(entry, identifier, f"DOES NOT RESOLVE ({error})", (str(error),), ())
    except LookupFailed as error:
        return Outcome(entry, identifier, "LOOKUP FAILED", (), (str(error),))
    status = (
        "verified" if not problems else f"resolves, {len(problems)} metadata problem(s)"
    )
    return Outcome(entry, identifier, status, tuple(problems), tuple(information))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis.verify_references",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tex",
        nargs="?",
        type=Path,
        default=None,
        metavar="TEX",
        help="manuscript source; not redistributed, so normally given explicitly",
    )
    return parser


def resolve_tex(cli: Path | None) -> Path:
    """Resolve the manuscript path, never relative to the current directory."""
    path = cli if cli is not None else DEFAULT_TEX
    if not path.is_file():
        raise SystemExit(
            f"error: manuscript source not found at '{path}'. The .tex is not "
            "redistributed in this repository; pass its path as the first argument."
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tex = resolve_tex(args.tex)
    entries = parse_entries(tex)
    print(f"{len(entries)} references in {tex.name}\n")
    print(f"{'key':<24}{'identifier':<40}{'status'}")
    print("-" * 104)

    outcomes: list[Outcome] = []
    for entry in entries:
        outcome = check(entry)
        outcomes.append(outcome)
        print(f"{outcome.entry.key:<24}{outcome.identifier[:39]:<40}{outcome.status}")
        if outcome.status != "NEEDS MANUAL CHECK":
            time.sleep(POLITE_DELAY_SECONDS)

    bad = [o for o in outcomes if o.problems]
    failed = [o for o in outcomes if o.status == "LOOKUP FAILED"]
    manual = [o for o in outcomes if o.status == "NEEDS MANUAL CHECK"]
    informative = [o for o in outcomes if o.information and o not in failed]

    if bad:
        print("\nMetadata problems:")
        for outcome in bad:
            for problem in outcome.problems:
                print(f"  {outcome.entry.key}: {problem}")
    if informative:
        print("\nInformational:")
        for outcome in informative:
            for note in outcome.information:
                print(f"  {outcome.entry.key}: {note}")
    if failed:
        print("\nCould not be checked (network or rate limiting):")
        for outcome in failed:
            print(f"  {outcome.entry.key}: {'; '.join(outcome.information)}")
    if manual:
        print("\nNo machine-checkable identifier (verify by hand):")
        for outcome in manual:
            print(f"  {outcome.entry.key}: {outcome.entry.text[:110]}")
            if outcome.entry.url:
                print(f"    url: {outcome.entry.url}")

    print(
        f"\nsummary: {len(entries)} entries, {len(bad)} with problems, "
        f"{len(failed)} unchecked, {len(manual)} manual"
    )
    if bad:
        return 1
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
