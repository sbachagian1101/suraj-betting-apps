"""Adaptive text-based horse-racing prediction engine.

The engine parses a pasted race table, detects current race context, builds
context-sensitive feature weights, scores every runner, and estimates win and
top-three probabilities with a deterministic Monte Carlo simulation.

This is a transparent heuristic scorecard. It is not a trained or calibrated
betting model and should be validated against historical results before money
is risked.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


APP_VERSION = "1.0.1"

FACTOR_LABELS: Dict[str, str] = {
    "recent_form": "Recent form",
    "record_12m": "12-month record",
    "career_record": "Career record",
    "distance_record": "Distance record",
    "course_record": "Course record",
    "course_distance": "Course-distance record",
    "going_suitability": "Going suitability",
    "surface_suitability": "Surface suitability",
    "jockey_rating": "Jockey rating",
    "trainer_rating": "Trainer rating",
    "fitness_dlr": "Fitness / days since run",
    "barrier_draw": "Barrier draw",
    "last_start": "Last-start performance",
    "class_move": "Class movement",
    "prize_proxy": "Prize-money / class proxy",
    "form_trend": "Form trend",
    "consistency": "Consistency",
}

BASE_WEIGHTS: Dict[str, float] = {
    "recent_form": 16.0,
    "record_12m": 8.0,
    "career_record": 5.0,
    "distance_record": 10.0,
    "course_record": 4.0,
    "course_distance": 4.0,
    "going_suitability": 8.0,
    "surface_suitability": 4.0,
    "jockey_rating": 5.0,
    "trainer_rating": 5.0,
    "fitness_dlr": 6.0,
    "barrier_draw": 5.0,
    "last_start": 7.0,
    "class_move": 3.0,
    "prize_proxy": 3.0,
    "form_trend": 4.0,
    "consistency": 3.0,
}

EXPECTED_HEADERS = [
    "Tab", "Horse", "Form L5", "BP", "12m%", "Car%", "Dist%", "DLR",
    "Crs%", "JRat", "TRat", "PM 12m", "GD%", "Turf%", "AW%", "SH%",
    "PM Car", "LS Det", "CD%",
]

HEADER_ALIASES = {
    "tab": "Tab",
    "horse": "Horse",
    "forml5": "Form L5",
    "formlast5": "Form L5",
    "bp": "BP",
    "barrier": "BP",
    "12m%": "12m%",
    "12m": "12m%",
    "car%": "Car%",
    "career%": "Car%",
    "dist%": "Dist%",
    "distance%": "Dist%",
    "dlr": "DLR",
    "crs%": "Crs%",
    "course%": "Crs%",
    "jrat": "JRat",
    "trat": "TRat",
    "pm12m": "PM 12m",
    "gd%": "GD%",
    "turf%": "Turf%",
    "aw%": "AW%",
    "sh%": "SH%",
    "pmcar": "PM Car",
    "lsdet": "LS Det",
    "cd%": "CD%",
}


@dataclass
class RecordTriple:
    win_pct: float = 0.0
    place_pct: float = 0.0
    starts: int = 0
    raw: str = ""

    @property
    def performance(self) -> float:
        # Wins are rewarded more heavily; place percentage adds consistency.
        return 0.65 * self.win_pct + 0.35 * self.place_pct


@dataclass
class LastStart:
    finish: Optional[int] = None
    margin_l: Optional[float] = None
    venue_code: str = ""
    distance_m: Optional[int] = None
    rating_high: Optional[int] = None
    class_number: Optional[int] = None
    odds: Optional[float] = None
    weight_kg: Optional[float] = None
    raw: str = ""


@dataclass
class Runner:
    tab: int
    horse: str
    form_l5: str
    barrier: Optional[int]
    record_12m: RecordTriple
    career: RecordTriple
    distance: RecordTriple
    dlr: Optional[int]
    course: RecordTriple
    jockey_rating: Optional[float]
    trainer_rating: Optional[float]
    prize_12m: float
    good: RecordTriple
    turf: RecordTriple
    aw: RecordTriple
    soft_heavy: RecordTriple
    prize_career: float
    last_start: LastStart
    course_distance: RecordTriple
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class RaceInfo:
    race_number: Optional[int] = None
    race_name: str = "Untitled Race"
    prize: Optional[float] = None
    race_type: str = ""
    class_code: str = ""
    class_number: Optional[int] = None
    rating_high: Optional[int] = None
    rating_low: Optional[int] = None
    time: str = ""
    distance_m: Optional[int] = None
    surface: str = "Turf"
    going: str = "Unknown"
    going_code: str = ""
    is_handicap: bool = False
    field_size: int = 0
    distance_band: str = "Unknown"


@dataclass
class FactorDetail:
    key: str
    label: str
    raw_value: str
    score: float
    weight: float
    contribution: float
    note: str
    available: bool = True


@dataclass
class Prediction:
    rank: int
    tab: int
    horse: str
    model_score: float
    win_pct: float
    top3_pct: float
    fair_odds: Optional[float]
    data_completeness: float
    uncertainty: float
    factor_details: List[FactorDetail]
    strengths: List[str]
    risks: List[str]


@dataclass
class RaceAnalysis:
    race: RaceInfo
    predictions: List[Prediction]
    weights: Dict[str, float]
    weight_notes: Dict[str, str]
    warnings: List[str]
    confidence_label: str
    confidence_score: float
    verdict: str
    exacta: List[str]
    trifecta: List[str]
    simulations: int
    parser_summary: str


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _clean_header(value: str) -> str:
    return re.sub(r"[^a-z0-9%]+", "", value.lower())


def _split_tabular_line(line: str) -> List[str]:
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t"))
    # Fallback for text copied with aligned multi-space columns.
    return re.split(r"\s{2,}", line.strip())


def _to_int(value: Any) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_money(value: str) -> float:
    parsed = _to_float(re.sub(r"[^0-9.\-]", "", value or ""))
    return max(0.0, parsed or 0.0)


def parse_record(value: str) -> RecordTriple:
    text = (value or "").strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*-\s*(\d+)", text)
    if not match:
        return RecordTriple(raw=text)
    return RecordTriple(
        win_pct=max(0.0, float(match.group(1))),
        place_pct=max(0.0, float(match.group(2))),
        starts=max(0, int(match.group(3))),
        raw=text,
    )


def _extract_class_number(value: str) -> Optional[int]:
    """Extract conventional race classes such as C5, CL4 or CLASS 3."""
    text = value or ""
    match = re.search(r"\b(?:CLASS|CL|C)\s*(\d+)", text, flags=re.I)
    return int(match.group(1)) if match else None


def parse_last_start(value: str) -> LastStart:
    text = (value or "").strip()
    result = LastStart(raw=text)
    if not text:
        return result

    # Common layouts include both plain finishes ("4-2L-...") and
    # ordinal finishes ("1st-0.8L-...", "2nd-5L-...").
    pieces = text.split("--")
    main = pieces[0].strip()
    match = re.match(
        r"^\s*(?P<finish>\d+)(?:st|nd|rd|th)?\s*-\s*"
        r"(?P<margin>\d+(?:\.\d+)?)L\s*-\s*"
        r"(?P<venue>[^-]+)\s*-\s*(?P<distance>\d+)m\s*-\s*(?P<rest>.+)$",
        main,
        flags=re.I,
    )
    if match:
        result.finish = int(match.group("finish"))
        result.margin_l = float(match.group("margin"))
        result.venue_code = match.group("venue").strip()
        result.distance_m = int(match.group("distance"))
        rest = match.group("rest")
        rating_match = re.search(r"RST\s*(\d+)", rest, flags=re.I)
        if rating_match:
            result.rating_high = int(rating_match.group(1))
        result.class_number = _extract_class_number(rest)

    if len(pieces) >= 2:
        result.odds = _to_float(re.sub(r"[^0-9.]", "", pieces[1]))
    if len(pieces) >= 3:
        result.weight_kg = _to_float(re.sub(r"[^0-9.]", "", pieces[2]))
    return result


def _extract_nonempty_cells(line: str) -> List[str]:
    return [cell.strip() for cell in _split_tabular_line(line) if cell.strip()]


def _detect_distance_band(distance_m: Optional[int]) -> str:
    if distance_m is None:
        return "Unknown"
    if distance_m <= 1200:
        return "Sprint"
    if distance_m <= 1700:
        return "Mile / short middle"
    if distance_m <= 2200:
        return "Middle distance"
    return "Staying"


def _detect_surface_and_going(line: str) -> Tuple[str, str, str]:
    upper = line.upper()
    surface = "Turf"
    if re.search(r"\b(?:AW|ALL\s*WEATHER|SYNTHETIC|POLYTRACK|TAPETA|PSF)\b", upper):
        surface = "Synthetic"
    elif re.search(r"\bDIRT\b", upper):
        surface = "Dirt"
    elif re.search(r"\bTURF\b", upper):
        surface = "Turf"

    going = "Unknown"
    code = ""
    explicit = [
        (r"GOOD\s*TO\s*FIRM|GOOD\s*FIRM", "Good to Firm"),
        (r"GOOD\s*TO\s*SOFT|GOOD\s*SOFT", "Good to Soft"),
        (r"SOFT\s*TO\s*HEAVY", "Soft to Heavy"),
        (r"\bHEAVY\b", "Heavy"),
        (r"\bSOFT\b", "Soft"),
        (r"\bYIELDING\b", "Yielding"),
        (r"\bGOOD\b", "Good"),
        (r"\bFIRM\b", "Firm"),
        (r"\bSTANDARD\b", "Standard"),
        (r"\bFAST\b", "Fast"),
        (r"\bSLOW\b", "Slow"),
    ]
    for pattern, label in explicit:
        if re.search(pattern, upper):
            going = label
            break

    # Compact codes used by the supplied source, e.g. "1650m, TURF S".
    code_match = re.search(
        r"\b(?:TURF|AW|ALL\s*WEATHER|SYNTHETIC|DIRT)\s+([A-Z]{1,3}(?:\d{1,2})?)\b",
        upper,
    )
    if code_match:
        code = code_match.group(1)
        if going == "Unknown":
            code_upper = code.upper()
            if code_upper.startswith("H"):
                going = "Heavy"
            elif code_upper.startswith("S") and code_upper not in {"STD", "ST"}:
                going = "Soft"
            elif code_upper.startswith("Y"):
                going = "Yielding"
            elif code_upper.startswith("F"):
                going = "Firm"
            elif code_upper.startswith("G"):
                going = "Good"
            elif code_upper in {"N", "STD", "ST"}:
                going = "Standard"

    if surface == "Synthetic" and going == "Unknown":
        going = "Standard"
    return surface, going, code


def parse_race_text(text: str) -> Tuple[RaceInfo, List[Runner], List[str], str]:
    """Parse race metadata and runner rows from a pasted table."""
    normalized = (text or "").replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    warnings: List[str] = []
    race = RaceInfo()

    # Metadata scan before the runner header.
    header_index: Optional[int] = None
    header_cells: List[str] = []
    for idx, line in enumerate(lines):
        cells = [c.strip() for c in _split_tabular_line(line)]
        cleaned = {_clean_header(c) for c in cells if c.strip()}
        if "tab" in cleaned and "horse" in cleaned and ("forml5" in cleaned or "bp" in cleaned):
            header_index = idx
            header_cells = cells
            break

    metadata_lines = lines[:header_index] if header_index is not None else lines
    for line in metadata_lines:
        nonempty = _extract_nonempty_cells(line)
        if not nonempty:
            continue
        joined = " ".join(nonempty)

        # First line often starts with race number followed by title and prize.
        if race.race_number is None and re.fullmatch(r"\d+", nonempty[0]):
            race.race_number = int(nonempty[0])
            if len(nonempty) >= 2:
                title = nonempty[1]
                prize_match = re.search(r"\[\s*([\d,]+(?:\.\d+)?)\s*\]", title)
                if prize_match:
                    race.prize = _parse_money(prize_match.group(1))
                    title = re.sub(r"\s*\[[^\]]+\]\s*", "", title).strip()
                race.race_name = title or race.race_name
            continue

        type_match = re.search(r"\bType\s*:\s*(.+)$", joined, flags=re.I)
        if type_match:
            race.race_type = type_match.group(1).strip()
            race.class_number = _extract_class_number(race.race_type)
            if race.class_number is not None:
                race.class_code = f"CL{race.class_number}"
            rating_match = re.search(r"RST\s*(\d+)\s*-\s*(\d+)", race.race_type, flags=re.I)
            if rating_match:
                race.rating_high = int(rating_match.group(1))
                race.rating_low = int(rating_match.group(2))
            continue

        time_match = re.fullmatch(r"\d{1,2}:\d{2}", nonempty[0])
        if time_match:
            race.time = nonempty[0]

        distance_match = re.search(r"\b(\d{3,4})\s*m\b", joined, flags=re.I)
        if distance_match:
            race.distance_m = int(distance_match.group(1))
            surface, going, code = _detect_surface_and_going(joined)
            race.surface, race.going, race.going_code = surface, going, code

    race.is_handicap = bool(re.search(r"\bHANDICAP\b|\bHCP\b", f"{race.race_name} {race.race_type}", flags=re.I))
    race.distance_band = _detect_distance_band(race.distance_m)

    if header_index is None:
        raise ValueError(
            "Runner table header not found. Paste the complete table including the row beginning with 'Tab' and 'Horse'."
        )

    # Build a flexible source-column to canonical-column map.
    canonical_by_index: Dict[int, str] = {}
    for idx, cell in enumerate(header_cells):
        alias = HEADER_ALIASES.get(_clean_header(cell))
        if alias:
            canonical_by_index[idx] = alias

    missing = [h for h in EXPECTED_HEADERS if h not in canonical_by_index.values()]
    if missing:
        warnings.append("Missing/unrecognised columns: " + ", ".join(missing))

    runners: List[Runner] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        cells = [c.strip() for c in _split_tabular_line(line)]
        first = next((c for c in cells if c.strip()), "")
        if not re.fullmatch(r"\d+", first):
            continue

        row: Dict[str, str] = {header: "" for header in EXPECTED_HEADERS}
        for idx, canonical in canonical_by_index.items():
            if idx < len(cells):
                row[canonical] = cells[idx].strip()

        tab = _to_int(row.get("Tab"))
        horse = (row.get("Horse") or "").strip()
        if tab is None or not horse:
            continue

        runners.append(
            Runner(
                tab=tab,
                horse=horse,
                form_l5=(row.get("Form L5") or "").strip(),
                barrier=_to_int(row.get("BP")),
                record_12m=parse_record(row.get("12m%", "")),
                career=parse_record(row.get("Car%", "")),
                distance=parse_record(row.get("Dist%", "")),
                dlr=_to_int(row.get("DLR")),
                course=parse_record(row.get("Crs%", "")),
                jockey_rating=_to_float(row.get("JRat")),
                trainer_rating=_to_float(row.get("TRat")),
                prize_12m=_parse_money(row.get("PM 12m", "")),
                good=parse_record(row.get("GD%", "")),
                turf=parse_record(row.get("Turf%", "")),
                aw=parse_record(row.get("AW%", "")),
                soft_heavy=parse_record(row.get("SH%", "")),
                prize_career=_parse_money(row.get("PM Car", "")),
                last_start=parse_last_start(row.get("LS Det", "")),
                course_distance=parse_record(row.get("CD%", "")),
                raw=row,
            )
        )

    if not runners:
        raise ValueError("No runner rows could be parsed beneath the header.")

    runners.sort(key=lambda r: r.tab)
    race.field_size = len(runners)
    if race.distance_m is None:
        warnings.append("Race distance was not detected; balanced distance weighting will be used.")
    if race.going == "Unknown":
        warnings.append("Going was not detected; general surface form will carry more weight.")
    if len(runners) < 3:
        warnings.append("Fewer than three runners were parsed.")

    parser_summary = (
        f"Parsed {len(runners)} runners | {race.distance_m or 'Unknown'}m | "
        f"{race.surface} {race.going} | {race.race_type or 'Type not detected'}"
    )
    return race, runners, warnings, parser_summary


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _minmax(values: Sequence[Optional[float]], neutral: float = 50.0, higher_better: bool = True) -> List[float]:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return [neutral] * len(values)
    lo, hi = min(valid), max(valid)
    if math.isclose(lo, hi):
        return [neutral if v is not None else neutral for v in values]
    result = []
    for value in values:
        if value is None or not math.isfinite(value):
            result.append(neutral)
            continue
        score = (value - lo) / (hi - lo) * 100.0
        result.append(score if higher_better else 100.0 - score)
    return result


def _record_adjusted_values(records: Sequence[RecordTriple], reliability_k: float) -> List[float]:
    observed = [record.performance for record in records if record.starts > 0]
    baseline = statistics.median(observed) if observed else 15.0
    adjusted: List[float] = []
    for record in records:
        reliability = record.starts / (record.starts + reliability_k) if record.starts > 0 else 0.0
        adjusted.append(baseline + reliability * (record.performance - baseline))
    return adjusted


def _form_metrics(form: str) -> Tuple[float, float, float, str]:
    # Most recent run is on the right in this source. x marks a spell/reset.
    positions: List[int] = []
    for char in (form or "").lower():
        if char.isdigit():
            positions.append(10 if char == "0" else int(char))
    if not positions:
        return 50.0, 50.0, 50.0, "No usable recent-form digits"

    point_map = {1: 100, 2: 84, 3: 72, 4: 62, 5: 54, 6: 47, 7: 40, 8: 34, 9: 28, 10: 20}
    recent = positions[-5:]
    weights = [0.70, 0.85, 1.00, 1.20, 1.45][-len(recent):]
    recent_score = sum(point_map[min(10, p)] * w for p, w in zip(recent, weights)) / sum(weights)

    if len(recent) >= 2:
        # Compare older half with newer half; lower finishing position is better.
        mid = max(1, len(recent) // 2)
        older = statistics.mean(recent[:mid])
        newer = statistics.mean(recent[mid:]) if recent[mid:] else recent[-1]
        trend = _clamp(50.0 + (older - newer) * 8.0)
        stdev = statistics.pstdev(recent)
        consistency = _clamp(100.0 - stdev * 13.0)
    else:
        trend, consistency = 50.0, 55.0

    note = f"Positions interpreted oldest→latest: {recent}"
    return recent_score, trend, consistency, note


def _fitness_score(dlr: Optional[int], distance_m: Optional[int]) -> Tuple[float, str, bool]:
    if dlr is None:
        return 50.0, "DLR missing", False
    # Slightly broader ideal window for longer races.
    ideal = 24.0 if (distance_m or 1600) <= 1700 else 30.0
    spread = 24.0 if (distance_m or 1600) <= 1700 else 30.0
    score = 100.0 * math.exp(-((dlr - ideal) / spread) ** 2)
    if dlr < 5:
        score *= 0.72
    elif dlr > 100:
        score *= 0.72
    return _clamp(score), f"{dlr} days since last run; ideal around {ideal:.0f} days", True


def _barrier_score(barrier: Optional[int], field_size: int) -> Tuple[float, str, bool]:
    if barrier is None or field_size <= 1:
        return 50.0, "Barrier missing", False
    score = 100.0 * (field_size - barrier) / max(1, field_size - 1)
    return _clamp(score), f"Barrier {barrier} of {field_size}", True


def _last_start_score(last: LastStart, current_distance: Optional[int]) -> Tuple[float, str, bool]:
    if last.finish is None:
        return 50.0, "Last-start detail unavailable", False

    finish_score = 100.0 * math.exp(-0.22 * max(0, last.finish - 1))
    margin = max(0.0, last.margin_l or 0.0)

    # For a winner, the margin is the winning margin, so a larger value is a
    # positive. For every other finisher, it is the beaten margin, so smaller
    # is better.
    if last.finish == 1:
        margin_score = _clamp(72.0 + 18.0 * math.sqrt(margin))
        margin_note = f"won by {margin:g}L"
    else:
        margin_score = 100.0 * math.exp(-0.22 * margin)
        margin_note = f"beaten {margin:g}L"

    if last.distance_m and current_distance:
        distance_similarity = 100.0 * math.exp(-abs(last.distance_m - current_distance) / 650.0)
    else:
        distance_similarity = 50.0

    score = 0.50 * finish_score + 0.32 * margin_score + 0.18 * distance_similarity
    note = f"Finished {last.finish}; {margin_note}"
    if last.distance_m:
        note += f" over {last.distance_m}m"
    return _clamp(score), note, True


def _class_move_score(last: LastStart, race: RaceInfo) -> Tuple[float, str, bool]:
    # Rating-band handicaps: a higher previous ceiling than today's ceiling is
    # treated as an easier assignment.
    if last.rating_high is not None and race.rating_high is not None:
        difference = last.rating_high - race.rating_high
        score = _clamp(50.0 + difference * 2.0)
        if difference > 0:
            note = f"Drops from RST{last.rating_high} level to RST{race.rating_high}"
        elif difference < 0:
            note = f"Rises from RST{last.rating_high} level to RST{race.rating_high}"
        else:
            note = f"Same RST{race.rating_high} level as last start"
        return score, note, True

    # Conventional classes: Class 1 is stronger than Class 2, and so on.
    if last.class_number is not None and race.class_number is not None:
        difference = race.class_number - last.class_number
        score = _clamp(50.0 + difference * 12.0)
        if difference > 0:
            note = f"Drops from CL{last.class_number} to CL{race.class_number}"
        elif difference < 0:
            note = f"Rises from CL{last.class_number} to CL{race.class_number}"
        else:
            note = f"Same CL{race.class_number} level as last start"
        return score, note, True

    return 50.0, "Comparable class or rating band unavailable", False


def _going_record(runner: Runner, race: RaceInfo) -> Tuple[RecordTriple, str]:
    going = race.going.lower()
    if race.surface == "Synthetic":
        return runner.aw, "All-weather record"
    if "heavy" in going or "soft" in going or "yield" in going or "slow" in going:
        return runner.soft_heavy, "Soft/heavy record"
    if "good" in going or "firm" in going or "fast" in going:
        return runner.good, "Good/firm record"
    return runner.turf if race.surface == "Turf" else runner.aw, "General surface record"


def _surface_record(runner: Runner, race: RaceInfo) -> Tuple[RecordTriple, str]:
    if race.surface == "Synthetic":
        return runner.aw, "All-weather record"
    return runner.turf, "Turf record"


def build_context_weights(race: RaceInfo) -> Tuple[Dict[str, float], Dict[str, str]]:
    multipliers = {key: 1.0 for key in BASE_WEIGHTS}
    reasons: Dict[str, List[str]] = {key: [] for key in BASE_WEIGHTS}

    def apply(key: str, multiplier: float, reason: str) -> None:
        multipliers[key] *= multiplier
        reasons[key].append(reason)

    distance = race.distance_m
    if distance is not None and distance <= 1200:
        apply("barrier_draw", 1.70, "Sprint: early position and draw carry extra influence")
        apply("jockey_rating", 1.15, "Sprint: tactical execution matters more")
        apply("recent_form", 1.10, "Sprint: current sharpness emphasised")
        apply("fitness_dlr", 1.08, "Sprint: race fitness emphasised")
        apply("career_record", 0.90, "Sprint: long-term record slightly reduced")
        apply("consistency", 0.90, "Sprint: tactical variance is higher")
    elif distance is not None and distance <= 1700:
        apply("barrier_draw", 1.25, "Mile/short-middle race: draw remains relevant")
        apply("distance_record", 1.15, "Distance-specific record emphasised")
        apply("recent_form", 1.05, "Current form receives a small uplift")
    elif distance is not None and distance <= 2200:
        apply("distance_record", 1.35, "Middle distance: proven stamina emphasised")
        apply("career_record", 1.15, "Middle distance: established ability matters more")
        apply("consistency", 1.15, "Middle distance: reliability emphasised")
        apply("fitness_dlr", 1.10, "Middle distance: preparation matters more")
        apply("barrier_draw", 0.75, "Middle distance: draw influence reduced")
    elif distance is not None:
        apply("distance_record", 1.55, "Staying race: distance record is a key factor")
        apply("career_record", 1.30, "Staying race: proven long-term ability emphasised")
        apply("consistency", 1.35, "Staying race: reliability and stamina emphasised")
        apply("fitness_dlr", 1.20, "Staying race: preparation receives extra weight")
        apply("trainer_rating", 1.10, "Staying race: conditioning role increased")
        apply("barrier_draw", 0.55, "Staying race: draw influence substantially reduced")
        apply("jockey_rating", 0.92, "Staying race: jockey rating slightly reduced versus stamina")

    going = race.going.lower()
    if "heavy" in going:
        apply("going_suitability", 1.80, "Heavy going: wet-track evidence strongly emphasised")
        apply("surface_suitability", 1.25, "Heavy going: turf suitability increased")
        apply("fitness_dlr", 1.25, "Heavy going: fitness demand increased")
        apply("distance_record", 1.20, "Heavy going: effective stamina requirement increased")
        apply("consistency", 1.30, "Heavy going: dependable profiles preferred")
        apply("career_record", 1.15, "Heavy going: proven horses receive a small uplift")
        apply("barrier_draw", 0.75, "Heavy going: draw is less predictable")
        apply("recent_form", 0.92, "Heavy going: generic recent form slightly reduced")
    elif any(token in going for token in ("soft", "yield", "slow")):
        apply("going_suitability", 1.45, "Soft/yielding going: wet-track record emphasised")
        apply("surface_suitability", 1.10, "Soft/yielding going: turf evidence increased")
        apply("fitness_dlr", 1.10, "Soft/yielding going: fitness demand increased")
        apply("distance_record", 1.10, "Soft/yielding going: stamina receives a small uplift")
        apply("consistency", 1.15, "Soft/yielding going: reliable runners preferred")
        apply("barrier_draw", 0.90, "Soft/yielding going: draw influence slightly reduced")
    elif any(token in going for token in ("good", "firm", "fast")):
        apply("barrier_draw", 1.05, "Good/firm going: track position is more dependable")
        apply("recent_form", 1.03, "Good/firm going: current speed/form slightly emphasised")

    if race.surface == "Synthetic":
        apply("surface_suitability", 1.55, "Synthetic surface: all-weather record is central")
        apply("course_distance", 1.20, "Synthetic surface: course-distance evidence increased")
        apply("course_record", 1.10, "Synthetic circuits can be specialist tracks")
        apply("barrier_draw", 1.20, "Synthetic circuits often make position/draw more important")
        apply("going_suitability", 0.72, "Synthetic surface: separate going factor reduced")

    if race.field_size >= 13:
        apply("barrier_draw", 1.25, "Large field: traffic and draw risk increased")
        apply("jockey_rating", 1.10, "Large field: tactical judgement increased")
        apply("consistency", 1.05, "Large field: dependable profiles preferred")
    elif race.field_size and race.field_size <= 7:
        apply("barrier_draw", 0.75, "Small field: draw influence reduced")
        apply("recent_form", 1.05, "Small field: current ability more directly expressed")

    if race.is_handicap:
        apply("class_move", 1.35, "Handicap: class movement receives extra weight")
        apply("career_record", 1.08, "Handicap: established rating ability slightly increased")
        apply("prize_proxy", 1.10, "Handicap: prize history used as a class proxy")
        apply("barrier_draw", 1.05, "Handicap: position can affect an evenly matched field")

    raw = {key: BASE_WEIGHTS[key] * multipliers[key] for key in BASE_WEIGHTS}
    total = sum(raw.values()) or 1.0
    weights = {key: value / total * 100.0 for key, value in raw.items()}
    notes = {
        key: "; ".join(reasons[key]) if reasons[key] else "Base balanced weighting"
        for key in BASE_WEIGHTS
    }
    return weights, notes


def _factor_raw_record(record: RecordTriple) -> str:
    return record.raw or f"{record.win_pct:g}-{record.place_pct:g}-{record.starts}"


def _create_factor_details(
    race: RaceInfo,
    runners: Sequence[Runner],
    weights: Dict[str, float],
    weight_notes: Dict[str, str],
    higher_rating_is_better: bool = True,
) -> Tuple[List[List[FactorDetail]], List[float], List[float]]:
    n = len(runners)

    forms = [_form_metrics(r.form_l5) for r in runners]
    recent_scores = [item[0] for item in forms]
    trend_scores = [item[1] for item in forms]
    consistency_scores = [item[2] for item in forms]

    record_12m_values = _record_adjusted_values([r.record_12m for r in runners], reliability_k=8.0)
    career_values = _record_adjusted_values([r.career for r in runners], reliability_k=18.0)
    distance_values = _record_adjusted_values([r.distance for r in runners], reliability_k=7.0)
    course_values = _record_adjusted_values([r.course for r in runners], reliability_k=7.0)
    cd_values = _record_adjusted_values([r.course_distance for r in runners], reliability_k=6.0)

    going_records = [_going_record(r, race) for r in runners]
    going_values = _record_adjusted_values([x[0] for x in going_records], reliability_k=5.0)
    surface_records = [_surface_record(r, race) for r in runners]
    surface_values = _record_adjusted_values([x[0] for x in surface_records], reliability_k=10.0)

    record_scores = {
        "record_12m": _minmax(record_12m_values),
        "career_record": _minmax(career_values),
        "distance_record": _minmax(distance_values),
        "course_record": _minmax(course_values),
        "course_distance": _minmax(cd_values),
        "going_suitability": _minmax(going_values),
        "surface_suitability": _minmax(surface_values),
    }

    jockey_scores = _minmax(
        [r.jockey_rating for r in runners], higher_better=higher_rating_is_better
    )
    trainer_scores = _minmax(
        [r.trainer_rating for r in runners], higher_better=higher_rating_is_better
    )
    rating_note = (
        "Higher source rating treated as stronger"
        if higher_rating_is_better
        else "Lower source rating treated as stronger"
    )

    prize_12m_scores = _minmax([math.log1p(r.prize_12m) for r in runners])
    prize_career_scores = _minmax([math.log1p(r.prize_career) for r in runners])
    prize_scores = [0.65 * a + 0.35 * b for a, b in zip(prize_12m_scores, prize_career_scores)]

    all_details: List[List[FactorDetail]] = []
    completeness_values: List[float] = []
    reliability_values: List[float] = []

    for idx, runner in enumerate(runners):
        fitness_score, fitness_note, fitness_available = _fitness_score(runner.dlr, race.distance_m)
        barrier_score, barrier_note, barrier_available = _barrier_score(runner.barrier, race.field_size)
        last_score, last_note, last_available = _last_start_score(runner.last_start, race.distance_m)
        class_score, class_note, class_available = _class_move_score(runner.last_start, race)

        factor_data: Dict[str, Tuple[float, str, str, bool]] = {
            "recent_form": (recent_scores[idx], runner.form_l5 or "—", forms[idx][3], bool(runner.form_l5)),
            "record_12m": (
                record_scores["record_12m"][idx],
                _factor_raw_record(runner.record_12m),
                f"Shrunk for sample size ({runner.record_12m.starts} starts)",
                runner.record_12m.starts > 0,
            ),
            "career_record": (
                record_scores["career_record"][idx],
                _factor_raw_record(runner.career),
                f"Career sample: {runner.career.starts} starts",
                runner.career.starts > 0,
            ),
            "distance_record": (
                record_scores["distance_record"][idx],
                _factor_raw_record(runner.distance),
                f"Distance sample: {runner.distance.starts} starts",
                runner.distance.starts > 0,
            ),
            "course_record": (
                record_scores["course_record"][idx],
                _factor_raw_record(runner.course),
                f"Course sample: {runner.course.starts} starts",
                runner.course.starts > 0,
            ),
            "course_distance": (
                record_scores["course_distance"][idx],
                _factor_raw_record(runner.course_distance),
                f"Course-distance sample: {runner.course_distance.starts} starts",
                runner.course_distance.starts > 0,
            ),
            "going_suitability": (
                record_scores["going_suitability"][idx],
                _factor_raw_record(going_records[idx][0]),
                f"{going_records[idx][1]}; {going_records[idx][0].starts} starts",
                going_records[idx][0].starts > 0,
            ),
            "surface_suitability": (
                record_scores["surface_suitability"][idx],
                _factor_raw_record(surface_records[idx][0]),
                f"{surface_records[idx][1]}; {surface_records[idx][0].starts} starts",
                surface_records[idx][0].starts > 0,
            ),
            "jockey_rating": (
                jockey_scores[idx],
                f"{runner.jockey_rating:.2f}" if runner.jockey_rating is not None else "—",
                rating_note,
                runner.jockey_rating is not None,
            ),
            "trainer_rating": (
                trainer_scores[idx],
                f"{runner.trainer_rating:.2f}" if runner.trainer_rating is not None else "—",
                rating_note,
                runner.trainer_rating is not None,
            ),
            "fitness_dlr": (
                fitness_score,
                str(runner.dlr) if runner.dlr is not None else "—",
                fitness_note,
                fitness_available,
            ),
            "barrier_draw": (
                barrier_score,
                str(runner.barrier) if runner.barrier is not None else "—",
                barrier_note,
                barrier_available,
            ),
            "last_start": (
                last_score,
                runner.last_start.raw or "—",
                last_note,
                last_available,
            ),
            "class_move": (
                class_score,
                (
                    f"Last RST{runner.last_start.rating_high}"
                    if runner.last_start.rating_high is not None
                    else f"Last CL{runner.last_start.class_number}"
                    if runner.last_start.class_number is not None
                    else "—"
                ),
                class_note,
                class_available,
            ),
            "prize_proxy": (
                prize_scores[idx],
                f"12m ${runner.prize_12m:,.0f}; career ${runner.prize_career:,.0f}",
                "Log-normalised within this field",
                runner.prize_12m > 0 or runner.prize_career > 0,
            ),
            "form_trend": (
                trend_scores[idx],
                runner.form_l5 or "—",
                "Improving recent finishing pattern scores higher",
                len(re.findall(r"\d", runner.form_l5 or "")) >= 2,
            ),
            "consistency": (
                consistency_scores[idx],
                runner.form_l5 or "—",
                "Lower variation in recent finishing positions scores higher",
                len(re.findall(r"\d", runner.form_l5 or "")) >= 2,
            ),
        }

        details: List[FactorDetail] = []
        available_weight = 0.0
        record_starts = []
        for key in BASE_WEIGHTS:
            score, raw_value, note, available = factor_data[key]
            weight = weights[key]
            contribution = score * weight / 100.0
            combined_note = note
            if weight_notes.get(key) and weight_notes[key] != "Base balanced weighting":
                combined_note += f" | Context: {weight_notes[key]}"
            details.append(
                FactorDetail(
                    key=key,
                    label=FACTOR_LABELS[key],
                    raw_value=raw_value,
                    score=round(score, 2),
                    weight=round(weight, 3),
                    contribution=round(contribution, 3),
                    note=combined_note,
                    available=available,
                )
            )
            if available:
                available_weight += weight

        # Reliability is based mainly on relevant empirical samples.
        record_starts.extend([
            runner.record_12m.starts,
            runner.career.starts,
            runner.distance.starts,
            runner.course.starts,
            runner.course_distance.starts,
            going_records[idx][0].starts,
            surface_records[idx][0].starts,
        ])
        reliability = statistics.mean(min(1.0, starts / 12.0) for starts in record_starts)
        completeness_values.append(_clamp(available_weight))
        reliability_values.append(reliability * 100.0)
        all_details.append(details)

    return all_details, completeness_values, reliability_values


def _confidence(analyses: Sequence[Prediction], field_size: int) -> Tuple[float, str]:
    if not analyses:
        return 0.0, "Low"
    top = analyses[0]
    second = analyses[1] if len(analyses) > 1 else top
    separation = _clamp((top.model_score - second.model_score) * 10.0)
    baseline = 100.0 / max(1, field_size)
    probability_edge = _clamp((top.win_pct - baseline) * 4.0)
    data_quality = statistics.mean(p.data_completeness for p in analyses[: min(3, len(analyses))])
    score = 0.34 * separation + 0.33 * probability_edge + 0.33 * data_quality
    label = "High" if score >= 68 else "Medium" if score >= 48 else "Low"
    return round(score, 1), label


def _build_verdict(race: RaceInfo, predictions: Sequence[Prediction], confidence: str) -> Tuple[str, List[str], List[str]]:
    if not predictions:
        return "No prediction available.", [], []
    top = predictions[0]
    dangers = predictions[1:3]
    danger_text = ", ".join(f"{p.tab} {p.horse}" for p in dangers) or "none"
    verdict = (
        f"Top selection: {top.tab} {top.horse} ({top.win_pct:.1f}% win, {top.top3_pct:.1f}% top-three). "
        f"Main dangers: {danger_text}. Overall model confidence: {confidence}."
    )
    exacta: List[str] = []
    trifecta: List[str] = []
    if len(predictions) >= 3:
        a, b, c = predictions[:3]
        exacta = [f"{a.tab}-{b.tab}", f"{a.tab}-{c.tab}", f"{b.tab}-{a.tab}"]
        trifecta = [
            f"{a.tab}/{b.tab},{c.tab}/{b.tab},{c.tab}",
            f"{a.tab},{b.tab}/{a.tab},{b.tab},{c.tab}/{a.tab},{b.tab},{c.tab}",
        ]
    return verdict, exacta, trifecta


def analyse_race_text(
    text: str,
    simulations: int = 25000,
    race_overrides: Optional[Dict[str, Any]] = None,
    higher_rating_is_better: bool = True,
    uncertainty_scale: float = 1.0,
) -> RaceAnalysis:
    """Score a race.

    ``uncertainty_scale`` multiplies the per-runner simulation noise. At 1.0
    the behaviour is exactly the original engine's. The noise standard
    deviation was never calibrated against results, and it is the only thing
    setting how confident the output looks: raising it spreads the win
    probabilities out without changing the ranking at all.
    """
    race, runners, warnings, parser_summary = parse_race_text(text)

    overrides = race_overrides or {}
    applied_overrides: List[str] = []
    if overrides.get("distance_m"):
        race.distance_m = int(overrides["distance_m"])
        applied_overrides.append(f"distance={race.distance_m}m")
    if overrides.get("surface"):
        race.surface = str(overrides["surface"])
        applied_overrides.append(f"surface={race.surface}")
    if overrides.get("going"):
        race.going = str(overrides["going"])
        applied_overrides.append(f"going={race.going}")
    if overrides.get("is_handicap") is not None:
        race.is_handicap = bool(overrides["is_handicap"])
        applied_overrides.append(f"handicap={'yes' if race.is_handicap else 'no'}")
    race.distance_band = _detect_distance_band(race.distance_m)
    if applied_overrides:
        warnings.append("Manual race-context override applied: " + ", ".join(applied_overrides))
        parser_summary = (
            f"Parsed {len(runners)} runners | {race.distance_m or 'Unknown'}m | "
            f"{race.surface} {race.going} | {race.race_type or 'Type not detected'}"
        )
    weights, weight_notes = build_context_weights(race)
    all_details, completeness_values, reliability_values = _create_factor_details(
        race, runners, weights, weight_notes, higher_rating_is_better
    )

    base_scores: List[float] = []
    uncertainties: List[float] = []
    for details, completeness, reliability in zip(all_details, completeness_values, reliability_values):
        score = sum(detail.contribution for detail in details)
        # Small transparent penalty for materially incomplete cards.
        score -= max(0.0, 78.0 - completeness) * 0.045
        base_scores.append(_clamp(score))
        uncertainties.append(
            (6.5 + (100.0 - completeness) * 0.055 + (100.0 - reliability) * 0.035)
            * max(float(uncertainty_scale), 0.05)
        )

    simulations = int(_clamp(float(simulations), 1000.0, 200000.0))
    wins = [0] * len(runners)
    top3 = [0] * len(runners)
    seed_material = text.encode("utf-8", errors="ignore") + str(simulations).encode("ascii")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    rng = random.Random(seed)

    for _ in range(simulations):
        performances = [
            base + rng.gauss(0.0, uncertainty)
            for base, uncertainty in zip(base_scores, uncertainties)
        ]
        order = sorted(range(len(runners)), key=performances.__getitem__, reverse=True)
        wins[order[0]] += 1
        for idx in order[: min(3, len(order))]:
            top3[idx] += 1

    preliminary: List[Prediction] = []
    for idx, runner in enumerate(runners):
        win_pct = wins[idx] / simulations * 100.0
        top3_pct = top3[idx] / simulations * 100.0
        sorted_strengths = sorted(all_details[idx], key=lambda d: d.contribution, reverse=True)
        strengths = [
            f"{detail.label} ({detail.score:.0f}/100)"
            for detail in sorted_strengths
            if detail.score >= 62
        ][:3]
        sorted_risks = sorted(all_details[idx], key=lambda d: d.score)
        risks = [
            f"{detail.label} ({detail.score:.0f}/100)"
            for detail in sorted_risks
            if detail.score <= 38
        ][:3]
        if not strengths:
            strengths = [f"Balanced profile ({base_scores[idx]:.1f} score)"]
        if not risks:
            risks = ["No major factor below 38/100"]

        preliminary.append(
            Prediction(
                rank=0,
                tab=runner.tab,
                horse=runner.horse,
                model_score=round(base_scores[idx], 2),
                win_pct=round(win_pct, 2),
                top3_pct=round(top3_pct, 2),
                fair_odds=round(100.0 / win_pct, 2) if win_pct > 0.05 else None,
                data_completeness=round(completeness_values[idx], 1),
                uncertainty=round(uncertainties[idx], 2),
                factor_details=all_details[idx],
                strengths=strengths,
                risks=risks,
            )
        )

    predictions = sorted(preliminary, key=lambda p: (p.win_pct, p.model_score), reverse=True)
    for rank, prediction in enumerate(predictions, start=1):
        prediction.rank = rank

    confidence_score, confidence_label = _confidence(predictions, race.field_size)
    verdict, exacta, trifecta = _build_verdict(race, predictions, confidence_label)

    return RaceAnalysis(
        race=race,
        predictions=predictions,
        weights=weights,
        weight_notes=weight_notes,
        warnings=warnings,
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        verdict=verdict,
        exacta=exacta,
        trifecta=trifecta,
        simulations=simulations,
        parser_summary=parser_summary,
    )


def analysis_to_dict(analysis: RaceAnalysis) -> Dict[str, Any]:
    return asdict(analysis)


def analysis_report_text(analysis: RaceAnalysis) -> str:
    race = analysis.race
    lines = [
        "HORSE RACING TEXT PREDICTOR",
        "=" * 72,
        f"Race: {race.race_number or '-'} - {race.race_name}",
        f"Time: {race.time or '-'} | Distance: {race.distance_m or '-'}m | Surface: {race.surface} | Going: {race.going}",
        f"Type: {race.race_type or '-'} | Field: {race.field_size} | Simulations: {analysis.simulations:,}",
        f"Confidence: {analysis.confidence_label} ({analysis.confidence_score:.1f}/100)",
        "",
        analysis.verdict,
        "",
        "RANKING",
        "-" * 72,
    ]
    for p in analysis.predictions:
        lines.append(
            f"{p.rank:>2}. #{p.tab:<2} {p.horse:<24} Score {p.model_score:>6.2f} | "
            f"Win {p.win_pct:>5.1f}% | Top 3 {p.top3_pct:>5.1f}% | Fair odds {p.fair_odds or '-'}"
        )
        lines.append(f"    Strengths: {', '.join(p.strengths)}")
        lines.append(f"    Risks: {', '.join(p.risks)}")

    if analysis.exacta:
        lines.extend(["", "Suggested exacta structures: " + ", ".join(analysis.exacta)])
    if analysis.trifecta:
        lines.append("Suggested trifecta structures: " + " | ".join(analysis.trifecta))
    if analysis.warnings:
        lines.extend(["", "WARNINGS"] + [f"- {warning}" for warning in analysis.warnings])

    lines.extend([
        "",
        "IMPORTANT",
        "This output is a transparent heuristic model, not a trained or calibrated guarantee. ",
        "Validate the scoring and probabilities against historical race results before betting.",
    ])
    return "\n".join(lines)
