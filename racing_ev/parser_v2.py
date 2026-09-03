"""Resilient wrapper around the original Enhanced Form parser.

Racing & Sports can reach Streamlit in several clipboard shapes. The original
parser is strongest on Markdown tables. This wrapper keeps that path unchanged,
then falls back to browser/TSV field rows and, finally, to the repeated runner
detail headers. The last fallback is intentionally conservative: it recovers
runner identity/number/price first, then delegates all profile and past-start
parsing back to the original parser.
"""
from __future__ import annotations

import re
from typing import Any

from . import parser as legacy

ParsedRace = legacy.ParsedRace


def _plain_cells(line: str) -> list[str]:
    """Split a browser clipboard row while preserving names with normal spaces."""
    s = line.strip()
    if not s:
        return []
    if "\t" in s:
        return [legacy._normalise_spaces(x) for x in s.split("\t")]
    # Browser clipboard exports sometimes use several spaces between table cells.
    return [legacy._normalise_spaces(x) for x in re.split(r"\s{2,}", s) if x.strip()]


def _row_odds(cells: list[str]) -> float | None:
    # Ignore the first cells because weights/ratings/barriers are numeric too.
    for cell in reversed(cells[2:]):
        clean = re.sub(r"[*_$£€R,]", "", cell).strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", clean):
            value = legacy._as_float(clean)
            if value is not None and value > 1.0:
                return value
    return None


def _runner_from_cells(cells: list[str], discipline: str) -> dict[str, Any] | None:
    if len(cells) < 3 or not re.fullmatch(r"\d{1,2}", cells[0]):
        return None
    tab = int(cells[0])
    if not 1 <= tab <= 30:
        return None
    name = legacy._link_text(cells[1]).upper()
    if not re.search(r"[A-Z]", name) or name in {"RUNNER", "HORSE", "GREYHOUND"}:
        return None
    scratch = any(re.search(r"\bSCR(?:ATCHED)?\b", c, re.I) for c in cells)
    out: dict[str, Any] = {
        "tab": tab,
        "runner": name,
        "scratched": scratch,
        "market_odds": None if scratch else _row_odds(cells),
    }
    if discipline == "thoroughbred":
        # Expected plain order: no, horse, WT, BP, jockey, JRat, trainer, TRat, prices...
        wt = re.findall(r"\d+(?:\.\d+)?", cells[2])
        out["weight_kg"] = float(wt[-1]) if wt else None
        out["barrier"] = legacy._as_int(cells[3]) if len(cells) > 3 else None
        out["draw"] = out["barrier"]
        out["jockey"] = cells[4].upper() if len(cells) > 4 else ""
        out["jockey_rating"] = legacy._as_float(cells[5]) if len(cells) > 5 else None
        out["trainer"] = cells[6].upper() if len(cells) > 6 else ""
        out["trainer_rating"] = legacy._as_float(cells[7]) if len(cells) > 7 else None
    elif discipline == "harness":
        out["hcp"] = cells[2] if len(cells) > 2 else ""
        out["driver"] = cells[3].upper() if len(cells) > 3 else ""
        out["trainer"] = cells[4].upper() if len(cells) > 4 else ""
        out["barrier"] = legacy._parse_harness_barrier(out["hcp"])
        out["handicap_metres"] = legacy._parse_handicap_metres(out["hcp"])
    else:
        out["weight_kg"] = legacy._as_float(cells[2]) if len(cells) > 2 else None
        out["trainer"] = cells[3].upper() if len(cells) > 3 else ""
        out["box"] = tab if tab <= 8 else None
    return out


def _parse_plain_field_rows(raw: str, discipline: str) -> list[dict[str, Any]]:
    """Read TSV/multi-space clipboard table rows."""
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in raw.splitlines():
        cells = _plain_cells(line)
        runner = _runner_from_cells(cells, discipline)
        if runner is None or runner["tab"] in seen:
            continue
        # Reject historical lines accidentally shaped like a field row. Current
        # rows appear before the Explanations/detail section.
        if "Explanations" in raw:
            summary_end = raw.find("Explanations")
            line_pos = raw.find(line)
            if line_pos > summary_end >= 0:
                continue
        rows.append(runner)
        seen.add(runner["tab"])
    return sorted(rows, key=lambda r: r["tab"])


def _detail_headers(cleaned: str) -> list[re.Match[str]]:
    # R&S runner profile: NAME 5yo B Gelding, NAME 4yo CH Mare, etc.
    # Keep the name lazy but require a colour/sex descriptor after age.
    pattern = re.compile(
        r"(?mi)^\s*(?P<name>[A-Z][A-Z0-9'&.()\-/ ]{1,70}?)\s+"
        r"(?P<age>\d+)yo\s+(?P<colour>[A-Z/]{1,8})\s+"
        r"(?P<sex>[A-Za-z][A-Za-z /-]{0,25}?)(?=\s+\(BP:|\s+\d+(?:\.\d+)?kg\b|\s*$)"
    )
    return list(pattern.finditer(cleaned))


def _nearest_tab(cleaned: str, header_start: int, used: set[int], default: int) -> int:
    prev = cleaned[max(0, header_start - 700):header_start]
    lines = [x.strip() for x in prev.splitlines() if x.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"\d{1,2}", line):
            value = int(line)
            if 1 <= value <= 30 and value not in used:
                return value
    while default in used:
        default += 1
    return default


def _summary_scratch_map(raw: str, names: list[str]) -> dict[str, bool]:
    """Best-effort scratch detection from the pre-Explanations field area."""
    end = raw.lower().find("explanations")
    summary = raw[:end] if end >= 0 else raw[: min(len(raw), 20000)]
    upper = summary.upper()
    ordered: list[tuple[int, str]] = []
    for name in names:
        idx = upper.find(name.upper())
        if idx >= 0:
            ordered.append((idx, name))
    ordered.sort()
    out = {name: False for name in names}
    for i, (start, name) in enumerate(ordered):
        stop = ordered[i + 1][0] if i + 1 < len(ordered) else min(len(summary), start + 1200)
        segment = summary[start:stop]
        out[name] = bool(re.search(r"\bSCR(?:ATCHED)?\b", segment, re.I))
    return out


def _recover_runners_from_details(raw: str, discipline: str) -> list[dict[str, Any]]:
    cleaned = legacy.clean_markdown(raw)
    matches = _detail_headers(cleaned)
    if not matches:
        return []

    # De-duplicate names while preserving profile order.
    unique: list[re.Match[str]] = []
    seen_names: set[str] = set()
    for match in matches:
        name = legacy._normalise_spaces(match.group("name")).upper()
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append(match)
    names = [legacy._normalise_spaces(m.group("name")).upper() for m in unique]
    scratch_map = _summary_scratch_map(raw, names)

    runners: list[dict[str, Any]] = []
    used_tabs: set[int] = set()
    for ordinal, match in enumerate(unique, start=1):
        name = legacy._normalise_spaces(match.group("name")).upper()
        tab = _nearest_tab(cleaned, match.start(), used_tabs, ordinal)
        used_tabs.add(tab)
        scratched = scratch_map.get(name, False)
        odds = legacy._current_odds_before_header(cleaned, match.start())
        runner: dict[str, Any] = {
            "tab": tab,
            "runner": name,
            "scratched": scratched,
            "market_odds": None if scratched else odds,
        }
        profile = legacy._parse_profile_header(match.group(0), discipline)
        if discipline == "thoroughbred":
            runner["barrier"] = profile.get("profile_barrier")
            runner["draw"] = runner["barrier"]
            runner["weight_kg"] = profile.get("profile_weight_kg")
        elif discipline == "greyhound":
            runner["box"] = tab if tab <= 8 else None
            runner["weight_kg"] = profile.get("profile_weight_kg")
        else:
            runner["hcp"] = ""
            runner["barrier"] = None
            runner["handicap_metres"] = 0
        runners.append(runner)
    return sorted(runners, key=lambda r: r["tab"])


def parse_field_table(raw: str, discipline: str | None = None) -> list[dict[str, Any]]:
    """Three-stage current field parser.

    1. Original Markdown table parser.
    2. Browser TSV / multi-space clipboard rows.
    3. Runner profile-header recovery.
    """
    discipline = discipline or legacy.detect_discipline(raw)
    runners = legacy.parse_field_table(raw, discipline)
    if runners:
        return runners
    runners = _parse_plain_field_rows(raw, discipline)
    if runners:
        return runners
    return _recover_runners_from_details(raw, discipline)


def parse_race_header(raw: str, discipline: str | None = None) -> dict[str, Any]:
    return legacy.parse_race_header(raw, discipline)


def parse_runner_details(raw: str, runners: list[dict[str, Any]], discipline: str | None = None):
    return legacy.parse_runner_details(raw, runners, discipline)


def parse_race(raw: str, discipline: str = "auto") -> ParsedRace:
    if not raw or len(raw.strip()) < 100:
        return ParsedRace(race={"discipline": "unknown"}, warnings=["Input is empty or too short."])
    resolved = legacy.detect_discipline(raw) if discipline == "auto" else discipline.lower()
    race = legacy.parse_race_header(raw, resolved)
    runners = parse_field_table(raw, resolved)
    warnings: list[str] = []
    if not runners:
        warnings.append(
            "No runners could be recovered. Paste the complete Enhanced Form page, "
            "including either the current field table or the individual runner profiles."
        )
        return ParsedRace(race=race, warnings=warnings)

    runners, histories, trials = legacy.parse_runner_details(raw, runners, resolved)
    active = [r for r in runners if not r.get("scratched")]
    race["field_size"] = len(active)
    race["declared_runners"] = len(runners)
    race["parser_mode"] = (
        "markdown-field-table"
        if legacy.parse_field_table(raw, resolved)
        else "browser-clipboard-fallback"
    )

    missing_odds = [
        r["runner"] for r in active
        if not r.get("market_odds") or float(r.get("market_odds") or 0) <= 1
    ]
    if missing_odds:
        warnings.append(
            "Runner data parsed, but current odds need review for: " + ", ".join(missing_odds)
            + ". Enter/correct those prices in the Prediction & EV tab."
        )
    missing_detail = [
        r["runner"] for r in active
        if not r.get("history_count") and not r.get("making_debut")
    ]
    if missing_detail:
        warnings.append("No recent-start history parsed for: " + ", ".join(missing_detail))
    return ParsedRace(race=race, runners=runners, histories=histories, trials=trials, warnings=warnings)
