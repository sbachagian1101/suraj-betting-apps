"""Robust Racing & Sports greyhound Enhanced Form parser.

Supports both Markdown-ish copies (including ChatGPT pasted pages) and ordinary
browser clipboard text. The field table supplies current runner / weight /
trainer / market data; detail blocks add form, trainer stats, box statistics,
track/distance records, and recent-run metrics.
"""
from __future__ import annotations

import re
from typing import Any


def _f(value: Any, default: float = 0.0) -> float:
    try:
        s = str(value).replace("$", "").replace(",", "").replace("sec", "").strip()
        return float(s)
    except Exception:
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().upper()


def _clean_md(text: str) -> str:
    """Remove common Markdown markup while preserving lines and table pipes."""
    text = str(text or "")
    text = (
        text.replace("\xa0", " ")
        .replace("\u2003", " ")
        .replace("\u2002", " ")
        .replace("\u202f", " ")
    )
    text = text.replace("\\-", "-").replace("\\:", ":").replace("\\&", "&")
    text = re.sub(r"\[([^\]]+)\]\([^\n]*?\)", r"\1", text)
    text = text.replace("***", "").replace("**", "").replace("__", "")
    text = re.sub(r"^[#>*]+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+(?=[A-Za-z\[])", "", text, flags=re.M)
    return "\n".join(line.strip() for line in text.splitlines())


def parse_header(raw: str) -> dict[str, Any]:
    t = _clean_md(raw)
    h: dict[str, Any] = {}
    m = re.search(r"(?mi)^(.+?)\s+Form Guide\s*\(Race\s*(\d+)\)", t)
    if m:
        h["track"], h["race_no"] = m.group(1).strip(), int(m.group(2))
    else:
        m = re.search(r"(?mi)^(.+?)\s+Form Guide", t)
        if m:
            h["track"] = m.group(1).strip()
    m = re.search(
        r"(?mi)^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(.+?\d{4})$",
        t,
    )
    if m:
        h["date"] = f"{m.group(1)}, {m.group(2)}"
    m = re.search(r"(?m)^(\d{1,2}:\d{2})\s*$", t)
    if m:
        h["time"] = m.group(1)
    m = re.search(r"(?is)\(local\)\s*\n+\s*([^\n]+?)\s*\n+\s*Type:", t)
    if m:
        h["race_name"] = m.group(1).strip()
    m = re.search(r"(?i)Type:\s*([^\n]+?)(?:\s+Fastest Time:|$)", t)
    if m:
        h["race_type"] = m.group(1).strip()
    m = re.search(r"(?i)Fastest Time:\s*([0-9:.]+)", t)
    if m:
        h["fastest_time"] = m.group(1)
    m = re.search(r"AUD\s*\$([\d,]+)", t, re.I)
    if m:
        h["prize"] = f"AUD ${m.group(1)}"
    m = re.search(
        r"(?mi)^(\d{3,4})m\s+([A-Z ]+?)\s+(FAST|GOOD|SLOW|WET|HEAVY)\s*$", t
    )
    if m:
        h["distance_m"] = int(m.group(1))
        h["surface"] = m.group(2).strip().upper()
        h["going"] = m.group(3).upper()
    return h


def _parse_summary_markdown(raw: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not re.match(r"^\|\s*\d{1,2}\s*\|", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 5:
            continue
        tab = _i(re.sub(r"\D", "", cells[0]))
        if not 1 <= tab <= 10:
            continue
        name_m = re.search(r"\[\**([^\]]+?)\**\]\(", cells[1])
        trainer_m = re.search(r"\[\**([^\]]+?)\**\]\(", cells[3])
        name = _clean_name(name_m.group(1) if name_m else re.sub(r"[\[\]*]", "", cells[1]))
        trainer = _clean_name(trainer_m.group(1) if trainer_m else re.sub(r"[\[\]*]", "", cells[3]))
        if not name or name.startswith("HTTP"):
            continue
        weight = _f(cells[2])
        scratch = any(re.search(r"\bscr\b", c, re.I) for c in cells[4:])
        odds = 999.0
        for c in reversed(cells[4:]):
            c2 = re.sub(r"[*$]", "", c).strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", c2):
                odds = _f(c2, 999.0)
                break
        if scratch:
            odds = 999.0
        out.append(
            {
                "tab": tab,
                "box": tab if tab <= 8 else 0,
                "horse": name,
                "weight": weight,
                "trainer": trainer,
                "scratched": scratch,
                "tab_odds": odds,
            }
        )
    return out


def _summary_segment_clean(raw: str) -> str:
    t = _clean_md(raw)
    start = re.search(r"(?mi)^\|?\s*TabRunnerWTTrainer\b.*$", t)
    if not start:
        start = re.search(r"(?mi)^Tab\s*Runner\s*WT\s*Trainer\b.*$", t)
    end = re.search(r"(?mi)^Explanations\s*$", t)
    if start:
        return t[start.end() : end.start() if end and end.start() > start.end() else len(t)]
    return ""


def _parse_summary_plain(raw: str) -> list[dict[str, Any]]:
    """Fallback for direct browser clipboard text."""
    t = _clean_md(raw)
    out: list[dict[str, Any]] = []

    for line in t.splitlines():
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
        if len(parts) < 4 or not re.fullmatch(r"\d{1,2}", parts[0]):
            continue
        tab = int(parts[0])
        if not 1 <= tab <= 10 or not re.search(r"[A-Za-z]", parts[1]):
            continue
        scratch = any(re.search(r"\bscr\b", p, re.I) for p in parts)
        weight_idx = next(
            (i for i, p in enumerate(parts[2:], start=2) if re.fullmatch(r"\d{2}(?:\.\d+)?", p)),
            None,
        )
        if weight_idx is None:
            continue
        weight = _f(parts[weight_idx])
        trainer = _clean_name(parts[weight_idx + 1] if weight_idx + 1 < len(parts) else "")
        odds = 999.0
        for p in reversed(parts[weight_idx + 2 :]):
            if re.fullmatch(r"\$?\d+(?:\.\d+)?", p):
                odds = _f(p, 999.0)
                break
        out.append(
            {
                "tab": tab,
                "box": tab if tab <= 8 else 0,
                "horse": _clean_name(parts[1]),
                "weight": weight,
                "trainer": trainer,
                "scratched": scratch,
                "tab_odds": 999.0 if scratch else odds,
            }
        )

    if out:
        return out

    seg = _summary_segment_clean(raw)
    if not seg:
        return []
    lines = [
        ln.strip()
        for ln in seg.splitlines()
        if ln.strip() and not set(ln.strip()) <= set("|- ")
    ]
    i = 0
    while i < len(lines):
        if not re.fullmatch(r"\d{1,2}", lines[i]):
            i += 1
            continue
        tab = int(lines[i])
        if not 1 <= tab <= 10 or i + 3 >= len(lines):
            i += 1
            continue
        name = _clean_name(lines[i + 1])
        if not re.search(r"[A-Z]", name) or not re.fullmatch(r"\d{2}(?:\.\d+)?", lines[i + 2]):
            i += 1
            continue
        weight = _f(lines[i + 2])
        trainer = _clean_name(lines[i + 3])
        j = i + 4
        scratch = False
        odds = 999.0
        while j < len(lines) and not re.fullmatch(r"\d{1,2}", lines[j]):
            if re.search(r"\bscr\b", lines[j], re.I):
                scratch = True
            elif re.fullmatch(r"\$?\d+(?:\.\d+)?", lines[j]):
                odds = _f(lines[j], 999.0)
            j += 1
        out.append(
            {
                "tab": tab,
                "box": tab if tab <= 8 else 0,
                "horse": name,
                "weight": weight,
                "trainer": trainer,
                "scratched": scratch,
                "tab_odds": 999.0 if scratch else odds,
            }
        )
        i = j
    return out


def parse_summary(raw: str) -> list[dict[str, Any]]:
    runners = _parse_summary_markdown(raw) or _parse_summary_plain(raw)
    seen: set[int] = set()
    dedup: list[dict[str, Any]] = []
    for r in runners:
        if r["tab"] not in seen:
            dedup.append(r)
            seen.add(r["tab"])
    scratched_boxes = sorted(
        r["tab"] for r in dedup if r["tab"] <= 8 and r.get("scratched")
    )
    reserve_active = [r for r in dedup if r["tab"] > 8 and not r.get("scratched")]
    for r, box in zip(reserve_active, scratched_boxes):
        r["box"] = box
        r["reserve_into_box"] = True
    return dedup


def _detail_header_matches(
    t: str, runners: list[dict[str, Any]]
) -> list[tuple[int, int, re.Match[str]]]:
    found: list[tuple[int, int, re.Match[str]]] = []
    for r in runners:
        pat = re.compile(
            rf"(?mi)^\s*{re.escape(r['horse'])}\s+(\d+)yo\b[^\n]*$"
        )
        matches = list(pat.finditer(t))
        if matches:
            found.append((matches[0].start(), r["tab"], matches[0]))
    return sorted(found, key=lambda x: x[0])


# Form figures line followed by a market line. The price may be a bare number,
# "$22", or bookmaker-prefixed as copied from the live page: "betfair$22" / "Tab$4.2".
_FORM_PRICE = re.compile(
    r"(?mi)^\s*([fFxX0-9]{3,10})\s*$\s*^\s*(?:betfair|tab)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)


def _runner_blocks(raw: str, runners: list[dict[str, Any]]) -> dict[int, str]:
    """Split blocks using `RUNNER NAME ...yo`, which is stable across copy formats."""
    t = _clean_md(raw)
    starts = _detail_header_matches(t, runners)
    blocks: dict[int, str] = {}
    for idx, (header_pos, tab, _) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(t)
        prefix_start = max(0, header_pos - 320)
        prefix = t[prefix_start:header_pos]
        pair_matches = list(_FORM_PRICE.finditer(prefix))
        start = prefix_start + pair_matches[-1].start() if pair_matches else header_pos
        blocks[tab] = t[start:end]
    return blocks


def _line_after_label(block: str, label: str) -> str:
    lines = [ln.strip() for ln in block.splitlines()]
    target = label.lower().strip()
    for i, ln in enumerate(lines[:-1]):
        if ln.lower().strip() == target:
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j]:
                    return lines[j]
    return ""


def _parse_record_value(value: str) -> tuple[str, int, int, int]:
    """Parse count mode (3-6-26) or WPS percentage mode (3%-29%-34)."""
    s = re.sub(r"\s+", "", value or "")
    m = re.search(r"(\d+(?:\.\d+)?)%-(\d+(?:\.\d+)?)%-(\d+)", s)
    if m:
        win_pct, top3_pct, starts = float(m.group(1)), float(m.group(2)), int(m.group(3))
        wins = int(round(starts * win_pct / 100.0))
        top3 = int(round(starts * top3_pct / 100.0))
        top3 = max(wins, min(top3, starts))
        p23 = max(top3 - wins, 0)
        return f"{m.group(1)}%-{m.group(2)}%-{starts}", wins, p23, starts
    m = re.search(r"(\d+)-(\d+)-(\d+)", s)
    if m:
        wins, p23, starts = map(int, m.groups())
        return f"{wins}-{p23}-{starts}", wins, p23, starts
    return "0-0-0", 0, 0, 0


_FILTER_LABELS = [
    "Car", "12m", "Crs", "Dist", "Crs & Dist",
    "FU", "2U", "3U", "ClockW", "AClockW", "Dirt", "Sand",
]

_RECORD_VALUE = r"(\d+(?:\.\d+)?%?\s*-\s*\d+(?:\.\d+)?%?\s*-\s*\d+)"


def _parse_filters(body: str) -> dict[str, str]:
    """Parse the Filters section even when values are glued to the next label.

    Live-page clipboard text flattens the WPS grid into runs like
    ``Car 17%-17%-612m 17%-17%-6Crs ...`` where ``17%-17%-6`` is Car's value and
    ``12m`` is the next label.  Each value is matched with a lookahead anchored on
    the remaining labels so greedy backtracking splits the digits correctly
    (``...-612m`` -> starts=6, label=12m; ``...-4212m`` -> starts=42).
    """
    m = re.search(r"(?is)\bFilters\b(.*?)(?:\bFacts\b|\Z)", body)
    seg = re.sub(r"\s+", " ", m.group(1) if m else body)
    out: dict[str, str] = {}
    cur = 0
    for i, label in enumerate(_FILTER_LABELS):
        followers = [re.escape(x) for x in _FILTER_LABELS[i + 1 :]]
        stop = r"(?=\s*(?:" + "|".join(followers + [r"Facts", r"\Z"]) + r"))"
        pat = re.compile(re.escape(label) + r"\s*" + _RECORD_VALUE + stop, re.I)
        got = pat.search(seg, cur)
        if not got:
            continue
        out[label] = got.group(1)
        cur = got.end()
    return out


def _parse_facts(body: str) -> dict[str, Any]:
    """Parse the Facts strip (also glue-tolerant: ``$196DLS 7DLW 20(3)ROI ...``)."""
    m = re.search(
        r"(?is)\bFacts\b(.*?)(?:Best Winning Times|Days Since Last Run|\Z)", body
    )
    seg = re.sub(r"\s+", " ", m.group(1) if m else body)
    d: dict[str, Any] = {}
    got = re.search(r"(?<![A-Za-z])DLS\s*:?\s*(\d+)", seg, re.I)
    if got:
        d["dls"] = int(got.group(1))
    got = re.search(r"(?<![A-Za-z])DLW\s*:?\s*(\d+)", seg, re.I)
    if got:
        d["dlw"] = int(got.group(1))
    got = re.search(r"\bROI\s*:?\s*([+-]?\d+(?:\.\d+)?)\s*%", seg, re.I)
    if got:
        d["roi"] = _f(got.group(1)) / 100.0
    got = re.search(r"\bCar PM\s*:?\s*\$([\d.,]+)\s*([kKmM]?)", seg, re.I)
    if got:
        val = _f(got.group(1))
        unit = got.group(2).lower()
        d["career_pm"] = val * 1_000_000 if unit == "m" else val * 1000 if unit == "k" else val
    got = re.search(r"\b12m PM\s*:?\s*\$([\d.,]+)\s*([kKmM]?)", seg, re.I)
    if got:
        val = _f(got.group(1))
        unit = got.group(2).lower()
        d["pm_12m"] = val * 1_000_000 if unit == "m" else val * 1000 if unit == "k" else val
    return d


def _time_seconds(value: str) -> float:
    """Convert '23.31' or '0:23.31' to seconds."""
    s = str(value).strip()
    if ":" in s:
        head, _, tail = s.rpartition(":")
        return _f(head) * 60.0 + _f(tail)
    return _f(s)


def _parse_box_stats(block: str) -> dict[int, dict[str, int]]:
    stats = {b: {"wins": 0, "places23": 0, "starts": 0} for b in range(1, 9)}
    lines = [ln.strip() for ln in block.splitlines()]
    for label, key in (("Win", "wins"), ("2nd/3rd", "places23"), ("Starts", "starts")):
        for ln in lines:
            if not re.match(rf"^\|?\s*{re.escape(label)}\b", ln, re.I):
                continue
            remainder = re.sub(
                rf"^\|?\s*{re.escape(label)}\s*\|?", "", ln, flags=re.I
            )
            nums = [int(x) for x in re.findall(r"(?<!\d)\d+(?!\d)", remainder)]
            if len(nums) >= 8:
                for b, val in enumerate(nums[:8], start=1):
                    stats[b][key] = val
            break
    return stats


def _parse_recent_runs(block: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in block.splitlines()]
    runs: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if "Margin" not in line or "Distance" not in line or "Runner Time" not in line:
            continue
        if "Trial Time" in line:
            continue
        r: dict[str, Any] = {}
        for j in range(max(0, i - 18), i):
            m = re.fullmatch(r"(\d+)\s+of\s+(\d+)", lines[j], re.I)
            if m:
                r["finish"], r["field"] = int(m.group(1)), int(m.group(2))
            m = re.fullmatch(
                r"(\d+)(?:st|nd|rd|th)\s*([+-]?\d+(?:\.\d+)?)",
                lines[j],
                re.I,
            )
            if m:
                r["mrk_rank"], r["mrk_delta"] = int(m.group(1)), _f(m.group(2))
        m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})", line)
        if m:
            r["date"] = m.group(1)
        m = re.search(r"([A-Za-z .'-]+)\s*\(AUSTRALIA\):", line, re.I)
        if m:
            r["track"] = m.group(1).strip().upper()
        specs = {
            "margin": r"\bMargin\s*([\d.]+)L",
            "distance": r"\bDistance\s*(\d+)m",
            "race_time": r"\bRace Time\s*([\d:.]+)",
            "bom": r"\bBOM\s*([\d:.]+)sec",
            "bom_adj": r"\bBOM Time Adj\s*([+-]?\d+(?:\.\d+)?)sec",
            "runner_time": r"\bRunner Time\s*([\d:.]+)",
            "first_split": r"\b1st Split\s*([\d.]+)",
            "prior_box": r"\bBP\s*(\d+)",
            "sp": r"\bSP\s*\$([\d.]+)",
            "weight": r"\bR\.WT\s*([\d.]+)kg",
        }
        for key, pat in specs.items():
            m = re.search(pat, line, re.I)
            if not m:
                continue
            if key in {"distance", "prior_box"}:
                r[key] = _i(m.group(1))
            elif ":" in m.group(1):
                r[key] = m.group(1)
            else:
                r[key] = _f(m.group(1))
        m = re.search(r"\bClass\s*(.*?)\s*Prize\b", line, re.I)
        if m:
            r["class"] = m.group(1).strip()
        m = re.search(
            r"\bStewards\s*(.*?)(?=\s*Inrunning Position|\s*Runner Sectional|\s*Race/Horse Sectionals|$)",
            line,
            re.I,
        )
        if m:
            r["stewards"] = m.group(1).strip().rstrip(".")
        m = re.search(
            r"Inrunning Position\s*(\d+)(?:st|nd|rd|th) Place on settling(?:\s+(\d+)(?:st|nd|rd|th) Place on turn)?",
            line,
            re.I,
        )
        if not m:
            m = re.search(
                r"(\d+)(?:st|nd|rd|th) Place on settling(?:\s+(\d+)(?:st|nd|rd|th) Place on turn)?",
                line,
                re.I,
            )
        if m:
            r["settle_pos"] = int(m.group(1))
            if m.group(2):
                r["turn_pos"] = int(m.group(2))
        m = re.search(r"L1m\s*\([\d.]+\s+(\d+)(?:st|nd|rd|th)\)", line, re.I)
        if m:
            r["split_rank"] = int(m.group(1))
        runs.append(r)
    return runs


def _parse_block(block: str, runner: dict[str, Any]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    # Scope: everything before the "NAME ...yo" header belongs to the block
    # prefix (form figures + market price); everything from the header onward is
    # this runner's own data.  Restricting stats searches to `body` prevents a
    # short previous block (e.g. a scratched dog) bleeding its box stats or
    # track/distance best time into this runner.
    hm = re.search(rf"(?mi)^\s*{re.escape(runner['horse'])}\s+(\d+)yo\b[^\n]*$", block)
    head = block[: hm.start()] if hm else block[:400]
    body = block[hm.start() :] if hm else block

    pair = _FORM_PRICE.search(head) or _FORM_PRICE.search(block[:400])
    if pair:
        d["form"] = pair.group(1).lower()
        d["bf_odds"] = _f(pair.group(2), 999.0)

    m = re.search(
        rf"(?mi)^\s*{re.escape(runner['horse'])}\s+(\d+)yo\s+([A-Z/ ]+?)\s+([A-Z])\s*$",
        body,
    )
    if m:
        d["age"], d["colour"], d["sex"] = int(m.group(1)), m.group(2).strip(), m.group(3)

    # Trainer last-50: label and value may be glued ("Tra L5014%-36%-50").
    m = re.search(
        r"(?i)Tra\s*L50\s*:?\s*(\d+(?:\.\d+)?)\s*%\s*-\s*(\d+(?:\.\d+)?)\s*%\s*-\s*(\d+)",
        body,
    )
    if not m:
        m = re.search(
            r"(\d+(?:\.\d+)?)%\s*-\s*(\d+(?:\.\d+)?)%\s*-\s*(\d+)",
            _line_after_label(body, "Tra L50"),
        )
    if m:
        d["trainer_win"] = float(m.group(1)) / 100.0
        d["trainer_place"] = float(m.group(2)) / 100.0
        d["trainer_l50_n"] = int(m.group(3))

    # Track/distance best time: same line, next line, or a dash for none.
    m = re.search(
        r"(?i)Tra/Dist\s*Best\s*Time\s*:?\s*(\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?)", body
    )
    if m:
        d["tra_dist_best"] = _time_seconds(m.group(1))
    else:
        value = _line_after_label(body, "Tra/Dist Best Time")
        if value and not re.fullmatch(r"[-–—\\]+", value.strip()):
            m = re.search(r"\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?", value)
            if m:
                d["tra_dist_best"] = _time_seconds(m.group(0))

    d["box_stats"] = _parse_box_stats(body)

    filters = _parse_filters(body)
    for label, key in (
        ("Car", "career"),
        ("12m", "12m"),
        ("Crs", "course"),
        ("Dist", "distance"),
        ("Crs & Dist", "course_distance"),
        ("FU", "fu"),
        ("2U", "2u"),
        ("3U", "3u"),
    ):
        value = filters.get(label) or _line_after_label(body, label)
        display, wins, p23, starts = _parse_record_value(value)
        d[f"{key}_rec"] = display
        d[f"{key}_wins"] = wins
        d[f"{key}_places23"] = p23
        d[f"{key}_starts"] = starts

    d.update(_parse_facts(body))
    for label, key in (("DLS", "dls"), ("DLW", "dlw")):
        if key in d:
            continue
        m = re.search(r"\d+", _line_after_label(body, label))
        if m:
            d[key] = int(m.group(0))

    # "Days Since Last Run: 279 days (6U)" is the most authoritative freshness
    # figure (it also survives spells) and carries runs-this-campaign.
    m = re.search(r"(?i)Days Since Last Run:\s*(\d+)\s*days(?:\s*\((\d+)U\))?", body)
    if m:
        d["dls"] = int(m.group(1))
        if m.group(2):
            d["runs_this_prep"] = int(m.group(2))

    d["recent_runs"] = _parse_recent_runs(body)
    return d


def parse(raw: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    header = parse_header(raw)
    runners = parse_summary(raw)
    if not runners:
        return header, [], ["Could not locate the greyhound runners table."]
    blocks = _runner_blocks(raw, runners)
    for r in runners:
        if r["tab"] in blocks:
            r.update(_parse_block(blocks[r["tab"]], r))
        else:
            warnings.append(f"No detail block found for #{r['tab']} {r['horse']}.")
        if r.get("scratched"):
            r["tab_odds"] = 999.0
            r["bf_odds"] = 999.0
        r.setdefault("form", "")
        r.setdefault("bf_odds", r.get("tab_odds", 999.0))
        r.setdefault("trainer_win", 0.10)
        r.setdefault("trainer_place", 0.35)
        r.setdefault("tra_dist_best", 0.0)
        r.setdefault("dls", 14)
        r.setdefault("recent_runs", [])
        r.setdefault(
            "box_stats",
            {b: {"wins": 0, "places23": 0, "starts": 0} for b in range(1, 9)},
        )
        for key in ("career", "12m", "course", "distance", "course_distance"):
            r.setdefault(f"{key}_rec", "0-0-0")
            r.setdefault(f"{key}_wins", 0)
            r.setdefault(f"{key}_places23", 0)
            r.setdefault(f"{key}_starts", 0)
    active = [r for r in runners if not r.get("scratched")]
    if any(r.get("box", 0) == 0 for r in active):
        warnings.append("One or more active reserves could not be assigned to a vacant box.")
    missing_detail = [r for r in active if r["tab"] not in blocks]
    if missing_detail:
        warnings.append(f"Detailed form was not parsed for {len(missing_detail)} active runner(s).")
    low_runs = [r for r in active if len(r.get("recent_runs", [])) < 3]
    if low_runs:
        warnings.append(
            f"{len(low_runs)} active runner(s) have fewer than three parsed recent runs; model confidence is reduced."
        )
    return header, runners, warnings
