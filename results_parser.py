"""Actual-result parser for one or many Bet365 meetings.

Supports the raw TAB-style text supplied by the user, simplified R1 labels,
partial results, and dead heats separated by commas.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

RaceResult = List[List[int]]
MeetingResultMap = Dict[str, Dict[int, RaceResult]]


def normalise_meeting_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper().replace("–", "-").replace("—", "-"))
    text = re.sub(r"^(?:CR|MR|BR|SRX?|PRX?|ORX?|ARM|ARO|SRO|PRB|MRX|SRP)\s+", "", text)
    text = text.replace(" - ", "-")
    aliases = {
        "SANDOWN-LAKESIDE": "SANDOWN-LAKESIDE",
        "SANDOWN -LAKESIDE": "SANDOWN-LAKESIDE",
        "SANDOWN- LAKESIDE": "SANDOWN-LAKESIDE",
        "SANDOWN LAKESIDE": "SANDOWN-LAKESIDE",
        "SUNSHINE COAST": "SUNSHINE COAST",
        "PORT HEADLAND": "PORT HEDLAND",
        "PORT HEDLAND": "PORT HEDLAND",
        "WYONG": "WYONG",
        "FORBES": "FORBES",
    }
    return aliases.get(text, text)


def parse_result_sequence(text: str) -> RaceResult:
    raw = str(text or "").strip().replace("–", "-").replace("—", "-")
    raw = re.sub(r"Fixed\s*Odds(?:Quaddie)?", "", raw, flags=re.I)
    raw = re.sub(r"Quaddie", "", raw, flags=re.I)
    raw = re.sub(r"\bRESULTS?\b\s*:?", "", raw, flags=re.I)
    raw = raw.strip(" :-\t")
    if not re.match(r"^\d+(?:,\d+)?(?:-\d+(?:,\d+)?)+$", raw):
        return []
    groups: RaceResult = []
    for token in raw.split("-"):
        nums = [int(value) for value in token.split(",") if value.strip().isdigit()]
        if nums:
            groups.append(nums)
    return groups


def _match_loaded_meeting(candidate: str, meeting_names: Sequence[str]) -> Optional[str]:
    key = normalise_meeting_name(candidate)
    mapping = {normalise_meeting_name(name): name for name in meeting_names}
    if key in mapping:
        return mapping[key]
    # Conservative fuzzy containment for prefixes such as meeting codes.
    for normal, original in mapping.items():
        if key and (key in normal or normal in key):
            return original
    return None


def parse_multi_meeting_results(text: str, meeting_names: Sequence[str]) -> MeetingResultMap:
    result: MeetingResultMap = {name: {} for name in meeting_names}
    current: Optional[str] = None
    sequential: Dict[str, int] = {name: 1 for name in meeting_names}
    lines = [line.strip() for line in str(text or "").replace("\r", "\n").splitlines() if line.strip()]
    for line in lines:
        explicit = re.match(r"^(?:(?:CR|MR|BR|SRX?|PRX?|ORX?|ARM|ARO|SRO|PRB|MRX|SRP)\s+)?(.+?)\s*$", line, flags=re.I)
        if explicit and not re.match(r"^\d", line):
            matched = _match_loaded_meeting(explicit.group(1), meeting_names)
            if matched:
                current = matched
                continue
        labelled = re.match(r"^(?:R|Race\s*)(\d+)\s*[:=]\s*(.+)$", line, flags=re.I)
        if labelled and current:
            parsed = parse_result_sequence(labelled.group(2))
            if parsed:
                result[current][int(labelled.group(1))] = parsed
            continue
        parsed = parse_result_sequence(line)
        if parsed and current:
            race_no = sequential[current]
            while race_no in result[current]:
                race_no += 1
            result[current][race_no] = parsed
            sequential[current] = race_no + 1
    return {meeting: rows for meeting, rows in result.items() if rows}


def flatten_result(result: RaceResult) -> List[int]:
    return [number for group in result for number in group]


def validate_result_for_race(race: Dict[str, Any], result: RaceResult) -> Tuple[bool, str]:
    active = {int(r["number"]) for r in race.get("runners", []) if r.get("status") == "ACTIVE"}
    if not result:
        return False, "No finishing positions supplied."
    flat = flatten_result(result)
    if len(flat) != len(set(flat)):
        return False, "A runner number is repeated in the result."
    missing = [number for number in flat if number not in active]
    if missing:
        return False, "Runner(s) not in the active field: " + ", ".join(map(str, missing))
    return True, "Valid partial/tied result."


def result_display(result: RaceResult) -> str:
    return "-".join(",".join(str(number) for number in group) for group in result)
