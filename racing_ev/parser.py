"""Tolerant parser for Racing & Sports Enhanced Form clipboard/Markdown text.

The source layout is not a formal API. This parser therefore uses stable labels
and runner names instead of brittle absolute column positions. It supports:

* thoroughbred racing (Australia, UK, France, South Africa style cards)
* harness racing (mobile starts and standing-start handicaps)
* Australian greyhound racing

The parser returns a race dictionary, one current-runner dictionary per runner,
and a flat list of historical starts. Missing optional fields are retained as
``None`` rather than causing a runner to be dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Iterable


NBSP_CHARS = "\xa0\u2002\u2003\u2009\u202f"
DATE_RE = re.compile(r"\b(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})\b")


def _normalise_spaces(value: str) -> str:
    value = str(value or "")
    for ch in NBSP_CHARS:
        value = value.replace(ch, " ")
    return re.sub(r"[ \t]+", " ", value).strip()


def clean_markdown(text: str) -> str:
    """Strip common Markdown while preserving lines and source labels."""
    text = str(text or "")
    for ch in NBSP_CHARS:
        text = text.replace(ch, " ")
    text = text.replace("\\|", "¦").replace("\\:", ":").replace("\\-", "-")
    # Keep link text. URLs are not model features and make label parsing noisy.
    text = re.sub(r"\[([^\]]+?)\]\([^\n]*?\)", r"\1", text)
    text = text.replace("***", "").replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*[#>]+\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+(?=[A-Za-z0-9\[])", "", text)
    text = text.replace("¦", "|")
    return "\n".join(_normalise_spaces(line) for line in text.splitlines())


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    s = _normalise_spaces(str(value))
    s = s.replace("$", "").replace(",", "").replace("sec", "").replace("kg", "")
    s = re.sub(r"^[+ ]+", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return int(f) if f is not None else None


def _time_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    s = str(value).strip().replace("sec", "")
    try:
        parts = [float(x) for x in s.split(":")]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _parse_record(value: str | None) -> dict[str, int | None]:
    """Parse Racing & Sports W-P-S strings such as ``4-6-31``."""
    if not value:
        return {"wins": None, "places": None, "starts": None}
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not m:
        return {"wins": None, "places": None, "starts": None}
    return {"wins": int(m.group(1)), "places": int(m.group(2)), "starts": int(m.group(3))}


def _parse_rate(value: str | None) -> dict[str, float | int | None]:
    """Parse ``win%-place%-sample`` strike-rate strings."""
    if not value:
        return {"win_rate": None, "place_rate": None, "sample": None}
    m = re.search(r"(\d+(?:\.\d+)?)%\s*-\s*(\d+(?:\.\d+)?)%\s*-\s*(\d+)", value)
    if not m:
        return {"win_rate": None, "place_rate": None, "sample": None}
    return {
        "win_rate": float(m.group(1)) / 100.0,
        "place_rate": float(m.group(2)) / 100.0,
        "sample": int(m.group(3)),
    }


def _split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row only on unescaped pipes."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return []
    cells = re.split(r"(?<!\\)\|", s[1:-1])
    return [_normalise_spaces(c.replace("\\|", "|")) for c in cells]


def _link_text(cell: str) -> str:
    m = re.search(r"\[\**([^\]]+?)\**\]\(", cell)
    if m:
        return _normalise_spaces(m.group(1))
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cell)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_]", "", value)
    return _normalise_spaces(value)


def detect_discipline(raw: str) -> str:
    lower = raw.lower()
    if "/form-guide/greyhound/" in lower or "greyhound form guide" in lower:
        return "greyhound"
    if "/form-guide/harness/" in lower or "harness form guide" in lower:
        return "harness"
    return "thoroughbred"


def _distance_to_metres(distance_text: str | None) -> int | None:
    if not distance_text:
        return None
    s = distance_text.lower().replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)m", s)
    if m:
        return int(round(float(m.group(1))))
    total = 0.0
    # UK/Irish notation: 1m2f, 6f, 2m4f110y.
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(m|f|y)", s):
        n = float(amount)
        if unit == "m":
            total += n * 1609.344
        elif unit == "f":
            total += n * 201.168
        else:
            total += n * 0.9144
    return int(round(total)) if total else None


def parse_race_header(raw: str, discipline: str | None = None) -> dict[str, Any]:
    discipline = discipline or detect_discipline(raw)
    t = clean_markdown(raw)
    out: dict[str, Any] = {"discipline": discipline}

    m = re.search(r"(?mi)^(.+?)\s+Form Guide\s*\(Race\s*(\d+)\)", t)
    if m:
        out["track"] = m.group(1).strip()
        out["race_no"] = int(m.group(2))

    m = re.search(
        r"(?mi)^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(.+?\d{4})$",
        t,
    )
    if m:
        out["date_text"] = f"{m.group(1)}, {m.group(2)}"
        for fmt in ("%A, %d %B %Y", "%A, %dth %B %Y", "%A, %dst %B %Y", "%A, %dnd %B %Y", "%A, %drd %B %Y"):
            try:
                out["date"] = datetime.strptime(out["date_text"], fmt).date().isoformat()
                break
            except ValueError:
                continue

    m = re.search(r"(?m)^(\d{1,2}:\d{2})\s*$", t)
    if m:
        out["time"] = m.group(1)

    # First meaningful line after '(local)' is the race title.
    m = re.search(r"(?is)\(local\)\s*\n+\s*([^\n]+)", t)
    if m:
        out["race_name"] = m.group(1).strip()

    # Race conditions usually sit on one line between title and prize.
    condition_line = ""
    if out.get("race_name"):
        idx = t.find(out["race_name"])
        tail = t[idx + len(out["race_name"]):]
        for line in tail.splitlines():
            if line and any(label in line for label in ("Age:", "Type:", "Fastest Time:", "WT:", "Sex:")):
                condition_line = line
                break
    if condition_line:
        for label, key in (("Age", "age_condition"), ("Sex", "sex_condition"), ("WT", "weight_condition"), ("Type", "race_type"), ("Fastest Time", "fastest_time_text")):
            m = re.search(rf"\b{re.escape(label)}:\s*(.*?)(?=\s+(?:Age|Sex|WT|Type|Fastest Time):|$)", condition_line, re.I)
            if m:
                out[key] = m.group(1).strip()
        if out.get("fastest_time_text"):
            mt = re.match(r"([0-9:.]+)", out["fastest_time_text"])
            if mt:
                out["fastest_time_s"] = _time_to_seconds(mt.group(1))

    # Find the standalone distance/surface/going line just before prize placings.
    lines = [ln for ln in t.splitlines() if ln]
    for line in lines:
        m = re.fullmatch(
            r"(?P<dist>(?:\d+(?:\.\d+)?m|(?:\d+m)?\d+(?:\.\d+)?f(?:\d+y)?|\d+m(?:\d+f)?(?:\d+y)?))\s+"
            r"(?P<surface>[A-Za-z ]+?)\s+(?P<going>FAST|GOOD|STANDARD|SLOW|SOFT|HEAVY(?:\s*\d+)?|YIELDING|GOOD TO SOFT|GOOD TO FIRM|FIRM|WET|N|F|G|S|Y|GS|GF|H(?:\s*\d+)?)",
            line,
            re.I,
        )
        if m:
            out["distance_text"] = m.group("dist")
            out["distance_m"] = _distance_to_metres(m.group("dist"))
            out["surface"] = m.group("surface").strip().upper()
            out["going"] = m.group("going").strip().upper()
            break

    # Prize/currency: use the first standalone total-prize-like amount after conditions.
    m = re.search(r"(?m)^\s*([A-Z]{3})\s*(?:£|€|R|\$)?\s*([\d,]+)\s*$", t)
    if not m:
        m = re.search(r"(?m)^\s*(GBP|EUR|AUD|ZAR)\s*[£€R$]?\s*([\d,]+)\s*$", t, re.I)
    if m:
        out["currency"] = m.group(1).upper()
        out["prize"] = _as_float(m.group(2))

    return out


def parse_field_table(raw: str, discipline: str | None = None) -> list[dict[str, Any]]:
    discipline = discipline or detect_discipline(raw)
    runners: list[dict[str, Any]] = []
    seen_tabs: set[int] = set()

    for line in raw.splitlines():
        cells = _split_markdown_row(line)
        if len(cells) < 4:
            continue
        tab_text = re.sub(r"\D", "", cells[0])
        if not tab_text or not re.fullmatch(r"\d{1,2}", tab_text):
            continue
        tab = int(tab_text)
        if tab in seen_tabs or tab < 1 or tab > 30:
            continue
        name = _link_text(cells[1]).upper()
        if not re.search(r"[A-Z]", name) or name in {"RUNNER", "HORSE"}:
            continue

        scratch = any(re.search(r"\bSCR(?:ATCHED)?\b", _link_text(c), re.I) for c in cells)
        numeric_candidates: list[float] = []
        for c in cells[2:]:
            c2 = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", c)
            c2 = re.sub(r"\[image\]\([^)]*\)", "", c2, flags=re.I)
            c2 = re.sub(r"[*_]", "", c2)
            c2 = _normalise_spaces(c2)
            if re.fullmatch(r"\$?\d+(?:\.\d+)?", c2):
                val = _as_float(c2)
                if val is not None:
                    numeric_candidates.append(val)
        # The odds are generally the right-most plain numeric field. Ratings occur
        # earlier, and scratched rows are forced to None.
        market_odds = numeric_candidates[-1] if numeric_candidates else None

        r: dict[str, Any] = {
            "tab": tab,
            "runner": name,
            "scratched": scratch,
            "market_odds": None if scratch else market_odds,
        }

        if discipline == "thoroughbred":
            wt_cell = _link_text(cells[2]) if len(cells) > 2 else ""
            nums = re.findall(r"\d+(?:\.\d+)?", wt_cell)
            r["weight_kg"] = float(nums[-1]) if nums else None
            r["barrier"] = _as_int(_link_text(cells[3])) if len(cells) > 3 else None
            r["jockey"] = _link_text(cells[4]).upper() if len(cells) > 4 else ""
            r["jockey_rating"] = _as_float(_link_text(cells[5])) if len(cells) > 5 else None
            r["trainer"] = _link_text(cells[6]).upper() if len(cells) > 6 else ""
            r["trainer_rating"] = _as_float(_link_text(cells[7])) if len(cells) > 7 else None
            r["draw"] = r["barrier"]
        elif discipline == "harness":
            r["hcp"] = _link_text(cells[2]) if len(cells) > 2 else ""
            r["driver"] = _link_text(cells[3]).upper() if len(cells) > 3 else ""
            r["trainer"] = _link_text(cells[4]).upper() if len(cells) > 4 else ""
            r["barrier"] = _parse_harness_barrier(r["hcp"])
            r["handicap_metres"] = _parse_handicap_metres(r["hcp"])
        else:
            r["weight_kg"] = _as_float(_link_text(cells[2])) if len(cells) > 2 else None
            r["trainer"] = _link_text(cells[3]).upper() if len(cells) > 3 else ""
            r["box"] = tab if tab <= 8 else None

        seen_tabs.add(tab)
        runners.append(r)

    return sorted(runners, key=lambda x: x["tab"])


def _parse_harness_barrier(value: str | None) -> int | None:
    if not value:
        return None
    m = re.fullmatch(r"(?:Fr|Sr)(\d+)", value.strip(), re.I)
    if not m:
        return None
    n = int(m.group(1))
    return n if value.lower().startswith("fr") else n + 6


def _parse_handicap_metres(value: str | None) -> int:
    if not value:
        return 0
    m = re.search(r"(\d+)\s*m", value, re.I)
    return int(m.group(1)) if m else 0


def _next_nonempty(lines: list[str], idx: int) -> str | None:
    for j in range(idx + 1, len(lines)):
        if lines[j].strip():
            return lines[j].strip()
    return None


def _label_value(block: str, label: str) -> str | None:
    lines = block.splitlines()
    target = label.lower().strip()
    for i, line in enumerate(lines):
        if line.lower().strip().rstrip(":") == target.rstrip(":"):
            return _next_nonempty(lines, i)
    return None


def _label_rate(block: str, anchor: str, stop_anchor: str | None = None) -> dict[str, Any]:
    low = block.lower()
    start = low.find(anchor.lower())
    if start < 0:
        return _parse_rate(None)
    end = low.find(stop_anchor.lower(), start + len(anchor)) if stop_anchor else -1
    segment = block[start : end if end > start else len(block)]
    m = re.search(r"\d+(?:\.\d+)?%\s*-\s*\d+(?:\.\d+)?%\s*-\s*\d+", segment)
    return _parse_rate(m.group(0) if m else None)


FILTER_LABELS = [
    "Car", "12m", "Crs", "Dist", "Crs & Dist", "Firm", "Good", "Soft", "Heavy",
    "AW", "Turf", "G1", "G2", "G3", "LR", "FU", "2U", "3U", "ClockW", "AClockW",
    "Dirt", "Sand",
]
FACT_LABELS = ["Car PM", "12m PM", "RTC/km", "RunsDistTC", "DLS", "DLW", "DOD", "ROI", "For-Against", "Hdle", "Stpl"]


def _parse_label_panel(block: str, start_label: str, end_labels: Iterable[str], labels: list[str]) -> dict[str, str]:
    lines = [ln.strip() for ln in block.splitlines()]
    try:
        start = next(i for i, ln in enumerate(lines) if ln.lower() == start_label.lower())
    except StopIteration:
        return {}
    end = len(lines)
    end_lower = {x.lower() for x in end_labels}
    for i in range(start + 1, len(lines)):
        if lines[i].lower() in end_lower:
            end = i
            break
    segment = lines[start + 1 : end]
    label_map = {x.lower(): x for x in labels}
    out: dict[str, str] = {}
    i = 0
    while i < len(segment):
        key = label_map.get(segment[i].lower())
        if key:
            j = i + 1
            while j < len(segment) and not segment[j]:
                j += 1
            if j < len(segment) and segment[j].lower() not in label_map:
                out[key] = segment[j]
                i = j + 1
                continue
        i += 1
    return out


def _find_detail_spans(cleaned: str, runners: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    starts: list[tuple[int, dict[str, Any]]] = []
    for runner in runners:
        name = re.escape(runner["runner"])
        # Detail headers have age/colour/sex after the runner name.
        pat = re.compile(rf"(?mi)^\s*{name}\s+\d+yo\b.*$")
        matches = list(pat.finditer(cleaned))
        if matches:
            # The last one is safest if a name appears in breadcrumbs or H2H.
            starts.append((matches[0].start(), runner))
    starts.sort(key=lambda x: x[0])
    spans: list[tuple[int, int, dict[str, Any]]] = []
    for i, (start, runner) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(cleaned)
        spans.append((start, end, runner))
    return spans


def _parse_box_stats(raw_block: str) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    lines = raw_block.splitlines()
    win_row = place_row = start_row = None
    for line in lines:
        cells = _split_markdown_row(line)
        if not cells:
            continue
        label = re.sub(r"[*_]", "", cells[0]).strip().lower()
        vals = [_as_int(re.sub(r"\D", "", c)) for c in cells[1:9]]
        vals = [v if v is not None else 0 for v in vals]
        if label == "win":
            win_row = vals
        elif label in {"2nd/3rd", "2nd / 3rd"}:
            place_row = vals
        elif label == "starts":
            start_row = vals
    if start_row:
        for box in range(1, min(8, len(start_row)) + 1):
            out[box] = {
                "wins": (win_row or [0] * 8)[box - 1],
                "places": (place_row or [0] * 8)[box - 1],
                "starts": start_row[box - 1],
            }
    return out


def _parse_profile_header(line: str, discipline: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"\b(\d+)yo\s+([A-Z/]+)\s+(.+?)(?=\s+\(BP:|\s+\d+(?:\.\d+)?kg|$)", line, re.I)
    if m:
        out["age"] = int(m.group(1))
        out["colour"] = m.group(2).upper()
        out["sex"] = m.group(3).strip()
    m = re.search(r"\(BP:\s*(\d+)\)", line, re.I)
    if m:
        out["profile_barrier"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)kg\b", line, re.I)
    if m:
        out["profile_weight_kg"] = float(m.group(1))
    return out


HISTORY_LABELS = [
    "Margin", "Distance", "Surface", "SOT", "Class", "Group", "Prize", "Prize Won", "API",
    "Race Time", "Sec Time", "BOM", "BOM Time Adj", "Runner Time", "1st Split", "R.WT",
    "Race Mile Rate", "Race Mile Rate Adj", "Jockey", "Driver", "Weight", "CD", "BP", "HCP",
    "SP", "Open Odds", "Trainer", "Ongoing Winners", "Rail", "Track Direction", "Gear Change",
    "Stewards", "Inrunning Position", "Tempo", "Runner Sectional", "Race/Horse Sectionals", "Video Comments",
]
HISTORY_LABEL_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(x) for x in HISTORY_LABELS), key=len, reverse=True)) + r")\b"
)


def _split_labeled_line(line: str) -> dict[str, str]:
    matches = list(HISTORY_LABEL_PATTERN.finditer(line))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        out[m.group(1)] = line[start:end].strip(" :")
    return out


def _parse_track_country(prefix: str, date_text: str) -> tuple[str | None, str | None]:
    tail = prefix[prefix.find(date_text) + len(date_text):]
    tail = re.sub(r"^\s*\([^)]*ago\)\s*", "", tail)
    tail = tail.strip(" :*-•")
    m = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", tail)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return tail or None, None


def _parse_history(block: str, runner_name: str, discipline: str) -> list[dict[str, Any]]:
    lines = block.splitlines()
    starts: list[dict[str, Any]] = []
    detail_indexes = [
        i for i, line in enumerate(lines)
        if DATE_RE.search(line) and HISTORY_LABEL_PATTERN.search(line)
    ]
    for pos, idx in enumerate(detail_indexes):
        line = lines[idx]
        dm = DATE_RE.search(line)
        if not dm:
            continue
        date_text = dm.group(1)
        labels = _split_labeled_line(line)
        first_label = HISTORY_LABEL_PATTERN.search(line)
        prefix = line[: first_label.start()] if first_label else line
        track, country = _parse_track_country(prefix, date_text)

        # R&S prints finish/field a few lines before the detailed date line.
        previous_idx = detail_indexes[pos - 1] if pos else max(-1, idx - 60)
        preceding = "\n".join(lines[previous_idx + 1:idx])
        finish = field_size = None
        finish_matches = list(re.finditer(r"(?mi)^\s*(\d+)\s+of\s+(\d+)\s*$", preceding))
        if finish_matches:
            fm = finish_matches[-1]
            finish, field_size = int(fm.group(1)), int(fm.group(2))
        dq = bool(re.search(r"(?mi)^\s*DQ[A-Z]*\s*$", preceding))
        rating_matches = list(re.finditer(
            r"(?mi)^\s*(\d+(?:\.\d+)?)\s*$\n\s*(OHR|IMR|MRK)\s*$", preceding
        ))
        rating_value = float(rating_matches[-1].group(1)) if rating_matches else None
        rating_type = rating_matches[-1].group(2).upper() if rating_matches else None

        run: dict[str, Any] = {
            "runner": runner_name,
            "discipline": discipline,
            "date": _parse_iso_date(date_text),
            "date_text": date_text,
            "track": track,
            "country": country,
            "finish": finish,
            "field_size": field_size,
            "disqualified": dq,
            "rating": rating_value,
            "rating_type": rating_type,
        }
        numeric_map = {
            "Margin": "margin",
            "Distance": "distance_m",
            "API": "api",
            "Race Time": "race_time_s",
            "Sec Time": "sectional_time_s",
            "BOM": "bom_time_s",
            "BOM Time Adj": "bom_time_adj_s",
            "Runner Time": "runner_time_s",
            "1st Split": "first_split_s",
            "R.WT": "runner_weight_kg",
            "Race Mile Rate": "mile_rate_s",
            "Race Mile Rate Adj": "mile_rate_adj",
            "Weight": "weight_kg",
            "BP": "barrier_or_box",
            "SP": "sp",
            "Open Odds": "open_odds",
            "HCP": "hcp_metres",
        }
        for label, key in numeric_map.items():
            value = labels.get(label)
            if value is None:
                continue
            if key.endswith("time_s") or key in {"race_time_s", "runner_time_s", "bom_time_s", "sectional_time_s", "mile_rate_s", "first_split_s"}:
                run[key] = _time_to_seconds(re.match(r"[0-9:.]+", value).group(0)) if re.match(r"[0-9:.]+", value) else _as_float(value)
            elif key == "distance_m":
                run[key] = _distance_to_metres(re.match(r"[0-9.]+m", value, re.I).group(0)) if re.match(r"[0-9.]+m", value, re.I) else _as_int(value)
            else:
                run[key] = _as_float(value)
        for label, key in {
            "Surface": "surface", "SOT": "going", "Class": "class", "Group": "group",
            "Jockey": "jockey", "Driver": "driver", "Trainer": "trainer", "Track Direction": "track_direction",
            "Gear Change": "gear_change", "Stewards": "stewards", "Inrunning Position": "inrunning",
            "Tempo": "tempo", "Video Comments": "video_comments", "Race/Horse Sectionals": "sectionals",
        }.items():
            if label in labels:
                run[key] = labels[label]
        if "Margin" in labels:
            run["margin"] = _as_float(labels["Margin"])
        # Derive useful runner-vs-race deltas.
        if run.get("runner_time_s") is not None and run.get("race_time_s") is not None:
            run["time_behind_s"] = float(run["runner_time_s"]) - float(run["race_time_s"])
        if run.get("bom_time_adj_s") is not None:
            run["bom_time_adj_s"] = _as_float(labels.get("BOM Time Adj"))
        starts.append(run)
    return starts


def _parse_iso_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%d %b %Y").date().isoformat()
    except ValueError:
        return None


def _parse_trials(block: str, runner_name: str, discipline: str) -> list[dict[str, Any]]:
    # Trial lines use BT and omit many race labels. Capture compactly for debutants.
    lines = block.splitlines()
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not DATE_RE.search(line) or "Trial Time" not in line:
            continue
        date_text = DATE_RE.search(line).group(1)
        labels = _split_labeled_line(line)
        preceding = "\n".join(lines[max(0, idx - 8):idx])
        place_match = list(re.finditer(r"(?mi)^\s*(\d+)(?:st|nd|rd|th)\s*$", preceding))
        place = int(place_match[-1].group(1)) if place_match else None
        first_label = HISTORY_LABEL_PATTERN.search(line)
        prefix = line[: first_label.start()] if first_label else line
        track, country = _parse_track_country(prefix, date_text)
        out.append({
            "runner": runner_name,
            "discipline": discipline,
            "date": _parse_iso_date(date_text),
            "track": track,
            "country": country,
            "trial_place": place,
            "distance_m": _distance_to_metres(labels.get("Distance")),
            "trial_time_s": _time_to_seconds((re.search(r"Trial Time\s+([0-9:.]+)", line, re.I) or [None, None])[1]),
            "jockey": labels.get("Jockey"),
            "driver": labels.get("Driver"),
            "trainer": labels.get("Trainer"),
        })
    return out


def _current_odds_before_header(cleaned: str, header_start: int) -> float | None:
    prev = cleaned[max(0, header_start - 350):header_start]
    candidates = re.findall(r"(?m)^\$?(\d+(?:\.\d+)?)\s*$", prev)
    if not candidates:
        return None
    return float(candidates[-1])


def parse_runner_details(raw: str, runners: list[dict[str, Any]], discipline: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    discipline = discipline or detect_discipline(raw)
    cleaned = clean_markdown(raw)
    spans = _find_detail_spans(cleaned, runners)
    histories: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []

    for start, end, runner in spans:
        block = cleaned[start:end]
        raw_block = raw  # box table lookup below is narrowed by name positions when possible.
        first_line = block.splitlines()[0] if block.splitlines() else ""
        runner.update(_parse_profile_header(first_line, discipline))
        if runner.get("profile_barrier") is not None and runner.get("barrier") is None:
            runner["barrier"] = runner["profile_barrier"]
        if runner.get("profile_weight_kg") is not None and runner.get("weight_kg") is None:
            runner["weight_kg"] = runner["profile_weight_kg"]

        detail_odds = _current_odds_before_header(cleaned, start)
        if runner.get("market_odds") is None and detail_odds is not None:
            runner["market_odds"] = detail_odds

        for label, key in (("Sire", "sire"), ("Dam", "dam"), ("Dam Sire", "dam_sire"), ("Win Dist.", "win_distance"), ("Raced Dist.", "raced_distance")):
            val = _label_value(block, label)
            if val:
                runner[key] = val

        if discipline == "thoroughbred":
            jockey = _label_value(block, "Jockey")
            trainer = _label_value(block, "Trainer")
            if jockey:
                runner["jockey"] = jockey.upper()
            if trainer:
                runner["trainer"] = trainer.upper()
            jr = _label_rate(block, "Jockey", "Trainer")
            tr = _label_rate(block, "Trainer", "Raced Dist.")
            runner.update({f"jockey_{k}": v for k, v in jr.items()})
            runner.update({f"trainer_{k}": v for k, v in tr.items()})
            runner["partnership_record"] = _label_value(block, "J/T")
            runner["runner_connection_record"] = _label_value(block, "J/H")
        elif discipline == "harness":
            driver = _label_value(block, "Driver")
            trainer = _label_value(block, "Trainer")
            if driver:
                runner["driver"] = driver.upper()
            if trainer:
                runner["trainer"] = trainer.upper()
            dr = _label_rate(block, "Driver", "Trainer")
            tr = _label_rate(block, "Trainer", "Raced Dist.")
            runner.update({f"driver_{k}": v for k, v in dr.items()})
            runner.update({f"trainer_{k}": v for k, v in tr.items()})
            runner["partnership_record"] = _label_value(block, "D/T")
            runner["runner_connection_record"] = _label_value(block, "D/H")
        else:
            trainer = _label_value(block, "Trainer")
            if trainer:
                runner["trainer"] = trainer.upper()
            rate = _parse_rate(_label_value(block, "Tra L50"))
            runner.update({f"trainer_{k}": v for k, v in rate.items()})
            runner["track_distance_best_s"] = _time_to_seconds(_label_value(block, "Tra/Dist Best Time"))

        filters = _parse_label_panel(block, "Filters", ["Facts"], FILTER_LABELS)
        for label, value in filters.items():
            rec = _parse_record(value)
            safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            runner[f"{safe}_wins"] = rec["wins"]
            runner[f"{safe}_places"] = rec["places"]
            runner[f"{safe}_starts"] = rec["starts"]

        facts = _parse_label_panel(block, "Facts", ["Spell", "Days Since Last Run", "Making Race Debut"], FACT_LABELS)
        runner["dls"] = _as_int(facts.get("DLS"))
        runner["roi_pct"] = _as_float(facts.get("ROI"))
        runner["for_against"] = facts.get("For-Against")
        runner["career_prizemoney"] = _as_float(facts.get("Car PM"))
        runner["runs_distance_this_campaign"] = _as_int(facts.get("RunsDistTC"))

        # Find run-of-preparation marker from Days Since Last Run line.
        m = re.search(r"Days Since Last Run:\s*(\d+)\s*days(?:\s*\(([^)]+)\))?", block, re.I)
        if m:
            runner["dls"] = int(m.group(1))
            runner["runup"] = m.group(2)
        runner["making_debut"] = bool(re.search(r"Making Race Debut", block, re.I))

        # Locate corresponding raw block for box tables. Header names are stable.
        raw_clean = clean_markdown(raw)
        raw_start = raw_clean.find(first_line)
        if raw_start >= 0:
            # Converting indexes between raw and clean is not exact; a broad name-to-next-name
            # extraction still safely contains only the current runner's first box table.
            raw_name_pat = re.compile(rf"(?mi)^.*{re.escape(runner['runner'])}.*\d+yo.*$")
            rm = raw_name_pat.search(raw)
            if rm:
                next_pos = len(raw)
                for other in runners:
                    if other is runner:
                        continue
                    om = re.compile(rf"(?mi)^.*{re.escape(other['runner'])}.*\d+yo.*$").search(raw, rm.end())
                    if om:
                        next_pos = min(next_pos, om.start())
                raw_block = raw[rm.start():next_pos]
        if discipline == "greyhound":
            box_stats = _parse_box_stats(raw_block)
            runner["box_stats"] = box_stats
            box = runner.get("box") or runner.get("tab")
            if box in box_stats:
                runner["current_box_wins"] = box_stats[box]["wins"]
                runner["current_box_places"] = box_stats[box]["places"]
                runner["current_box_starts"] = box_stats[box]["starts"]

        h = _parse_history(block, runner["runner"], discipline)
        histories.extend(h)
        runner["history_count"] = len(h)
        t = _parse_trials(block, runner["runner"], discipline)
        trials.extend(t)
        runner["trial_count"] = len(t)

    # Reserve greyhounds can replace scratched boxes; apply after all scratch states known.
    if discipline == "greyhound":
        scratched_boxes = [r["tab"] for r in runners if r.get("scratched") and r["tab"] <= 8]
        reserves = [r for r in runners if not r.get("scratched") and r["tab"] > 8]
        for reserve, box in zip(reserves, sorted(scratched_boxes)):
            reserve["box"] = box
            reserve["reserve_into_box"] = True

    return runners, histories, trials


@dataclass
class ParsedRace:
    race: dict[str, Any]
    runners: list[dict[str, Any]] = field(default_factory=list)
    histories: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def discipline(self) -> str:
        return str(self.race.get("discipline", "unknown"))


def parse_race(raw: str, discipline: str = "auto") -> ParsedRace:
    if not raw or len(raw.strip()) < 100:
        return ParsedRace(race={"discipline": "unknown"}, warnings=["Input is empty or too short."])
    resolved = detect_discipline(raw) if discipline == "auto" else discipline.lower()
    race = parse_race_header(raw, resolved)
    runners = parse_field_table(raw, resolved)
    warnings: list[str] = []
    if not runners:
        warnings.append("No runners were found in the field table.")
        return ParsedRace(race=race, warnings=warnings)
    runners, histories, trials = parse_runner_details(raw, runners, resolved)
    active = [r for r in runners if not r.get("scratched")]
    race["field_size"] = len(active)
    race["declared_runners"] = len(runners)
    missing_odds = [r["runner"] for r in active if not r.get("market_odds") or r.get("market_odds", 0) <= 1]
    if missing_odds:
        warnings.append("Missing or invalid current odds for: " + ", ".join(missing_odds))
    missing_detail = [r["runner"] for r in active if not r.get("history_count") and not r.get("making_debut")]
    if missing_detail:
        warnings.append("No recent-start history parsed for: " + ", ".join(missing_detail))
    return ParsedRace(race=race, runners=runners, histories=histories, trials=trials, warnings=warnings)
