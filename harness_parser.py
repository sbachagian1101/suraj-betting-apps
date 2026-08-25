"""Parser for Racing & Sports harness Enhanced Form pages.

Design note
-----------
The field table is read via its **header row**, never by column position, and
rows keep their empty cells when split.  The previous parser accepted only
markdown pipe-table rows (`| 1 | ASTERI ... |`), so a plain select-all/copy from
the live page - which is tab separated - produced **zero** runners.

Harness pages differ from the thoroughbred ones in ways that matter:

* the field table carries **Driver** and money columns (`Tot $PM`, `Dri L50`),
  no weight and no barrier column;
* each recent run carries **IMR** (individual mile rate) and often an **OHR**
  (official handicap rating) as a bare number on the line *before* the literal
  `OHR`;
* the Filters panel has no going categories (Firm/Good/Soft/Heavy) - harness
  uses AW/Turf/Dirt/Sand instead;
* start position appears as `HCP Fr5` / `HCP Sr2` (front or second row) inside
  the results line rather than as a barrier column.
"""
from __future__ import annotations

import re
from typing import Any

BOOKMAKERS = (
    "betfair", "bet365", "tab", "tabtouch", "ladbrokes", "sportsbet",
    "neds", "unibet", "pointsbet", "bluebet", "boombet", "palmerbet", "usr", "grs",
)
_NAME_HEADERS = ("runner", "horse", "greyhound")
_PRICE_CELL = re.compile(r"^\$?(\d+(?:\.\d+)?)$")
_PCT_TRIPLE = re.compile(r"(\d+(?:\.\d+)?)%-(\d+(?:\.\d+)?)%-(\d+)")
_WPS = re.compile(r"(\d+)-(\d+)-(\d+)")
_SEXES = "Colt|Gelding|Filly|Mare|Horse|Stallion|Ridgling|Rig"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(round(_f(value, default)))
    except (TypeError, ValueError):
        return default


def _secs(value: str, default: float = 0.0) -> float:
    """`2:00.24` -> 120.24 seconds. Mile rates and race times use M:SS.ss."""
    m = re.match(r"^\s*(\d+):(\d{2}(?:\.\d+)?)\s*$", str(value or ""))
    if not m:
        return _f(value, default)
    return int(m.group(1)) * 60 + float(m.group(2))


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .-\t")


def _clean_md(text: str) -> str:
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    if "|--" in t:                       # a genuine markdown table
        t = re.sub(r"[ \t]*\|[ \t]*", "\t", t)
    return t


# --------------------------------------------------------------------------
# race header
# --------------------------------------------------------------------------
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
        r"(?mi)^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(.+?\d{4})$", t)
    if m:
        h["date"] = f"{m.group(1)}, {m.group(2)}"
    m = re.search(r"(?m)^(\d{1,2}:\d{2})\s*$", t)
    if m:
        h["time"] = m.group(1)
    m = re.search(r"(?is)\(local\)\s*\n+\s*([^\n]+?)\s*\n+\s*(?:Age|Type):", t)
    if m:
        h["race_name"] = m.group(1).strip()
    # Harness pages label the class line `Age:`, thoroughbred pages use `Type:`.
    m = re.search(r"(?i)\b(?:Age|Type):\s*([^\n]+?)(?:\s+Fastest Time:|$)", t)
    if m:
        h["race_type"] = m.group(1).strip()
    m = re.search(r"(?i)Fastest Time:\s*([0-9:.]+)", t)
    if m:
        h["fastest_time"] = m.group(1)
    m = re.search(r"AUD\s*\$([\d,]+)", t, re.I)
    if m:
        h["prize"] = f"AUD ${m.group(1)}"
    m = re.search(
        r"(?mi)^(\d{3,4})m\s+([A-Z][A-Z ]*?)\s+(FIRM|GOOD|SOFT|HEAVY|FAST|SLOW|WET|SYNTHETIC)\s*(\d+)?\s*$",
        t)
    if m:
        h["distance_m"] = int(m.group(1))
        h["surface"] = m.group(2).strip().upper()
        h["going"] = m.group(3).upper()
        if m.group(4):
            h["going_rating"] = int(m.group(4))
    return h


# --------------------------------------------------------------------------
# field table
# --------------------------------------------------------------------------
def _split_row(line: str) -> list[str]:
    """Split a field-table row, keeping empty cells so columns stay aligned."""
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    return [p.strip() for p in re.split(r"\s{2,}", line)]


def _summary_header_map(cells: list[str]) -> dict[str, Any] | None:
    low = [c.strip().lower() for c in cells]
    if len(low) < 3 or low[0] != "tab":
        return None
    name = next((i for i, c in enumerate(low) if c in _NAME_HEADERS), None)
    if name is None:
        return None

    def find(*labels: str) -> int | None:
        return next((i for i, c in enumerate(low) if c in labels), None)

    trainer = find("trainer")
    anchor = trainer if trainer is not None else name
    return {
        "name": name,
        "driver": find("driver", "reinsman", "jockey"),
        "trainer": trainer,
        "total_pm": find("tot $pm", "tot$pm", "total $pm"),
        "pm_per_start": find("car$/st", "car $/st"),
        # Bookmaker columns to the right of the trainer, in page order. Located
        # by header so the money columns (`Dri L50`, `Tra PM`) can never be
        # mistaken for a price.
        "prices": [i for i, c in enumerate(low) if i > anchor and c in BOOKMAKERS],
        "anchor": anchor,
    }


def _parse_summary_header_driven(t: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    colmap: dict[str, Any] | None = None
    for line in t.splitlines():
        cells = _split_row(line)
        if colmap is None:
            colmap = _summary_header_map(cells)
            continue
        if not cells or not re.fullmatch(r"\d{1,2}", cells[0]):
            continue
        tab = int(cells[0])
        name_idx = colmap["name"]
        name = _clean_name(cells[name_idx]) if name_idx < len(cells) else ""
        if not 1 <= tab <= 30 or not re.search(r"[A-Za-z]", name):
            continue

        def cell(idx: int | None) -> str:
            return cells[idx] if idx is not None and idx < len(cells) else ""

        scratch = any(re.fullmatch(r"scr(atched)?", c, re.I) for c in cells)
        odds = 999.0
        idxs = colmap["prices"] or list(range(colmap["anchor"] + 1, len(cells)))
        for i in reversed(idxs):
            m = _PRICE_CELL.match(cell(i))
            if m:
                odds = _f(m.group(1), 999.0)
                break
        out.append({
            "tab": tab,
            # WA harness fields draw front-line runners by tab number, so the
            # tab doubles as the gate. `HCP Fr/Sr` in past runs is historical.
            "gate": tab,
            "horse": _clean_name(name).upper(),
            "driver": _clean_name(cell(colmap["driver"])).upper(),
            "trainer": _clean_name(cell(colmap["trainer"])).upper(),
            "total_pm": _f(re.sub(r"[^\d.]", "", cell(colmap["total_pm"]))) * (
                1000 if "k" in cell(colmap["total_pm"]).lower() else 1),
            "pm_per_start": _f(re.sub(r"[^\d.]", "", cell(colmap["pm_per_start"]))) * (
                1000 if "k" in cell(colmap["pm_per_start"]).lower() else 1),
            "scratched": scratch,
            "tab_odds": 999.0 if scratch else odds,
        })
    return out


def parse_summary(raw: str) -> list[dict[str, Any]]:
    runners = _parse_summary_header_driven(_clean_md(raw))
    seen: set[int] = set()
    dedup: list[dict[str, Any]] = []
    for r in runners:
        if r["tab"] not in seen:
            dedup.append(r)
            seen.add(r["tab"])
    return dedup


# --------------------------------------------------------------------------
# detail blocks
# --------------------------------------------------------------------------
def _horse_header_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?mi)^\s*{re.escape(name)}\s+(\d+)yo\b[^\n]*$")


# Form figures, then a bookmaker price. The bookmaker token can carry a pipe
# ("USR|GRS$15"), and some runners have no price line at all.
_FORM_PRICE = re.compile(
    r"(?mi)^\s*([0-9xX]{1,10})\s*$\s*^\s*([A-Za-z][A-Za-z0-9|/ ]*)?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)
_FORM_ONLY = re.compile(r"(?mi)^\s*([0-9xX]{1,10})\s*$")


def _runner_blocks(raw: str, runners: list[dict[str, Any]]) -> dict[int, str]:
    t = _clean_md(raw)
    found: list[tuple[int, int]] = []
    for r in runners:
        m = _horse_header_re(r["horse"]).search(t)
        if m:
            found.append((m.start(), r["tab"]))
    found.sort()
    # Resolve every block's start first. A block ends where the *next block*
    # starts, not where the next header line is: the form-figures/price pair
    # sits above its own header, so ending at the header would let one runner's
    # block swallow the next runner's form and price.
    starts: list[tuple[int, int]] = []
    for pos, tab in found:
        prefix_start = max(0, pos - 220)
        prefix = t[prefix_start:pos]
        pairs = list(_FORM_PRICE.finditer(prefix))
        if pairs:
            start = prefix_start + pairs[-1].start()
        else:                                   # no price line at all
            solo = list(_FORM_ONLY.finditer(prefix))
            start = prefix_start + solo[-1].start() if solo else pos
        starts.append((start, tab))
    blocks: dict[int, str] = {}
    for idx, (start, tab) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(t)
        blocks[tab] = t[start:end]
    return blocks


def _glued_panel(seg: str, labels: list[str]) -> dict[str, str]:
    """Read an R&S glued panel where each line is `<value><next label>`.

        Car
        0-2-312m      -> Car = "0-2-3", next label "12m"

    Labels that begin with a digit ("12m") make a pure regex ambiguous, so the
    panel is walked in order, peeling the known next label off each line's end.
    """
    known = sorted(labels, key=len, reverse=True)
    lines = [ln.strip() for ln in seg.splitlines() if ln.strip()]
    out: dict[str, str] = {}
    current: str | None = None
    for ln in lines:
        if current is None:
            if ln in labels:
                current = ln
            continue
        suffix = next((lb for lb in known if ln.endswith(lb) and len(ln) > len(lb)), None)
        if suffix is None:
            if ln in labels:
                current = ln
                continue
            out[current] = ln
            current = None
            continue
        out[current] = ln[: -len(suffix)].strip()
        current = suffix
    return out


# Harness filter panels carry surface categories, not going categories.
_FILTER_LABELS = ["Car", "12m", "Crs", "Dist", "Crs & Dist", "AW", "Turf",
                  "G1", "G2", "G3", "LR", "FU", "2U", "3U",
                  "ClockW", "AClockW", "Dirt", "Sand"]
_FACT_LABELS = ["Car PM", "12m PM", "DLS", "DLW", "DOD", "ROI",
                "For-Against", "Hdle", "Stpl"]


def _parse_filters(body: str) -> dict[str, tuple[int, int, int]]:
    m = re.search(r"(?is)\bFilters.*?(?=\nFacts\b|\nDays Since Last Run|\nSpell\b|\Z)", body)
    if not m:
        return {}
    out: dict[str, tuple[int, int, int]] = {}
    for label, value in _glued_panel(m.group(0), _FILTER_LABELS).items():
        g = _WPS.fullmatch(value.strip())
        if g:
            out[label] = (int(g.group(1)), int(g.group(2)), int(g.group(3)))
    return out


def _parse_facts(body: str) -> dict[str, str]:
    m = re.search(r"(?is)\nFacts\b.*?(?=\nDays Since Last Run|\nSpell\b|\Z)", body)
    if not m:
        return {}
    return _glued_panel(m.group(0), _FACT_LABELS)


def _segment(body: str, start_pat: str, end_pats: list[str]) -> str:
    m = re.search(start_pat, body, re.M | re.I)
    if not m:
        return ""
    rest = body[m.end():]
    stops = [e.start() for e in
             (re.search(pat, rest, re.M | re.I) for pat in end_pats) if e]
    return rest[:min(stops)] if stops else rest


def _pct_in(seg: str, label: str) -> tuple[float, float, int]:
    """`Last5022%-54%-50` inside a segment -> (0.22, 0.54, 50).

    Anchored on the owning label rather than counting occurrences: a runner with
    no declared driver has an *empty* Last50 line, and counting would hand the
    trainer's strike rate to the driver.
    """
    m = re.search(re.escape(label) + r"\s*" + _PCT_TRIPLE.pattern, seg, re.I)
    if not m:
        return 0.0, 0.0, 0
    return _f(m.group(1)) / 100.0, _f(m.group(2)) / 100.0, int(m.group(3))


_IMR_LINE = re.compile(r"^\s*(?:(\d+)(?:st|nd|rd|th)\s+)?(\d+:\d{2}(?:\.\d+)?)\s*$")


def _parse_recent_runs(block: str) -> list[dict[str, Any]]:
    """Parse completed race lines. Barrier trials (`BT Results`) are skipped."""
    lines = block.splitlines()
    runs: list[dict[str, Any]] = []
    for i, ln in enumerate(lines):
        fin = re.fullmatch(r"\s*(\d{1,2}) of (\d{1,2})\s*", ln)
        if not fin:
            continue
        window = lines[i + 1: i + 22]
        res = ""
        for j, w in enumerate(window):
            if re.match(r"\s*\d{1,2} \w{3} \d{4} \(", w):
                res = w
                window = window[:j]          # IMR/OHR sit above the results line
                break
        if not res:
            continue
        run: dict[str, Any] = {"finish": int(fin.group(1)), "field": int(fin.group(2))}

        # IMR: the line after the literal `IMR`, optionally rank-prefixed.
        for k, w in enumerate(window):
            if w.strip().upper() == "IMR" and k + 1 < len(window):
                m = _IMR_LINE.match(window[k + 1])
                if m:
                    if m.group(1):
                        run["imr_rank"] = int(m.group(1))
                    run["imr"] = _secs(m.group(2))
                break
        # OHR: the bare number on the line *before* the literal `OHR`.
        for k, w in enumerate(window):
            if w.strip().upper() == "OHR" and k >= 1:
                prev = window[k - 1].strip()
                if re.fullmatch(r"\d{1,3}", prev):
                    run["ohr"] = int(prev)
                break

        def grab(pat: str, cast=_f, default=None):
            m = re.search(pat, res, re.I)
            return cast(m.group(1)) if m else default

        run["date"] = grab(r"^\s*(\d{1,2} \w{3} \d{4})", str, "")
        run["days_ago"] = grab(r"\((\d+)d ago\)", _i, None)
        run["track"] = grab(r"\)\s+([A-Z][A-Za-z' ]+?)\s+\(", str, "").strip()
        run["margin"] = grab(r"Margin\s+([\d.]+)L")
        run["distance"] = grab(r"Distance\s+(\d+)m", _i)
        run["surface"] = grab(r"Surface\s+(\w+)", str, "")
        run["race_class"] = grab(r"Class\s+(.+?)\s+(?:Group|Prize)", str, "")
        run["group"] = grab(r"Group\s+(\w+)", str, "")
        run["api"] = grab(r"\bAPI\s+([\d.]+)")
        run["race_time"] = _secs(grab(r"Race Time\s+([\d:.]+)", str, "") or "")
        run["mile_rate"] = _secs(grab(r"Race Mile Rate\s+([\d:.]+)", str, "") or "")
        run["mile_rate_adj"] = grab(r"Race Mile Rate Adj\s+([+-]?[\d.]+)")
        run["sp"] = grab(r"\bSP\s+\$([\d.]+)")
        run["driver"] = _clean_name(grab(r"Driver\s+([A-Z][A-Za-z' .-]+?)\s+HCP", str, ""))
        hcp = grab(r"\bHCP\s+((?:Fr|Sr)\d+)", str, "")
        if hcp:
            run["hcp"] = hcp.upper()
            run["second_row"] = hcp.upper().startswith("SR")
            g = re.search(r"(\d+)$", hcp)
            run["prior_gate"] = int(g.group(1)) if g else 0
        run["bell_pos"] = grab(r"(\d+)(?:st|nd|rd|th) position at Bell Lap", _i)
        m = re.search(r"Stewards\s+(.+?)(?:\s+Inrunning|\s+Video Comments|$)", res, re.I)
        run["stewards"] = m.group(1).strip() if m else ""
        runs.append(run)
    return runs


def _parse_block(block: str, runner: dict[str, Any]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    hm = _horse_header_re(runner["horse"]).search(block)
    head = block[: hm.start()] if hm else block[:300]
    body = block[hm.start():] if hm else block

    pair = _FORM_PRICE.search(head)
    if pair:
        d["form"] = pair.group(1).lower()
        book = (pair.group(2) or "").strip().lower()
        price = _f(pair.group(3), 999.0)
        d["price_source"] = book or "unknown"
        if "betfair" in book:
            d["bf_odds"] = price
        elif book in ("tab", "tabtouch"):
            if _f(runner.get("tab_odds"), 999.0) >= 999.0:
                d["tab_odds"] = price
            d["book_odds"] = price
        else:
            d["book_odds"] = price          # Ladbrokes, USR|GRS, ...
    else:
        solo = _FORM_ONLY.search(head)
        if solo:
            d["form"] = solo.group(1).lower()

    if hm:
        m = re.match(rf"(?i)^\s*.+?\s+(\d+)yo\s+([A-Z/ ]+?)\s+({_SEXES})\s*$", hm.group(0).strip())
        if m:
            d["age"] = _i(m.group(1))
            d["colour"] = m.group(2).strip()
            d["sex"] = m.group(3)

    for key, label in (("sire", "Sire"), ("dam", "Dam"), ("dam_sire", "Dam Sire")):
        m = re.search(rf"(?mi)^{label}([A-Z][^\n]*)$", body)
        if m:
            d[key] = _clean_name(m.group(1))

    driver_seg = _segment(body, r"^Driver", [r"^Trainer", r"^Raced Dist\."])
    trainer_seg = _segment(body, r"^Trainer", [r"^Raced Dist\.", r"^D/H", r"^Win Dist\."])
    d["driver_win"], d["driver_place"], d["driver_l50_n"] = _pct_in(driver_seg, "Last50")
    d["trainer_win"], d["trainer_place"], d["trainer_l50_n"] = _pct_in(trainer_seg, "Last50")
    d["driver_horse_win"], d["driver_horse_place"], d["driver_horse_n"] = _pct_in(body, "D/H")
    d["driver_trainer_win"], d["driver_trainer_place"], d["driver_trainer_n"] = _pct_in(body, "D/T")

    m = re.search(r"(?i)Raced Dist\.\s*(\d+)m\s*-\s*(\d+)m", body)
    if m:
        d["dist_min"], d["dist_max"] = _i(m.group(1)), _i(m.group(2))
    m = re.search(r"(?i)Win Dist\.\s*(\d+)m", body)
    if m:
        d["win_dist"] = _i(m.group(1))

    m = re.search(r"(?i)Days Since Last Run:\s*(\d+)\s*days", body)
    if m:
        d["dls"] = _i(m.group(1))

    filt = _parse_filters(body)
    d["filters"] = filt
    for label, key in (("Car", "career"), ("12m", "m12"), ("Crs", "course"),
                       ("Dist", "distance"), ("Crs & Dist", "course_distance"),
                       ("AW", "aw"), ("Turf", "turf"), ("FU", "first_up"),
                       ("2U", "second_up"), ("3U", "third_up")):
        w, p, s = filt.get(label, (0, 0, 0))
        d[f"{key}_wins"], d[f"{key}_places"], d[f"{key}_starts"] = w, p, s

    facts = _parse_facts(body)
    d["facts"] = facts
    d["maiden"] = str(facts.get("DLW", "")).lower().startswith("maiden")

    runs = _parse_recent_runs(body)
    d["recent_runs"] = runs
    if runs:
        d["last_fin"] = runs[0]["finish"]
        ohrs = [r["ohr"] for r in runs if r.get("ohr")]
        if ohrs:
            d["latest_ohr"] = ohrs[0]
        imrs = [r["imr"] for r in runs if r.get("imr")]
        if imrs:
            d["best_imr"] = min(imrs)
    return d


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
def parse(raw: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    header = parse_header(raw)
    runners = parse_summary(raw)
    if not runners:
        return header, [], ["Could not locate the harness runners field table."]

    blocks = _runner_blocks(raw, runners)
    for r in runners:
        if r["tab"] in blocks:
            r.update(_parse_block(blocks[r["tab"]], r))
        if r.get("scratched"):
            r["tab_odds"] = 999.0
            r["bf_odds"] = 999.0
        r.setdefault("bf_odds", r.get("book_odds") or r.get("tab_odds", 999.0))
        r.setdefault("form", "")
        r.setdefault("dls", 21)
        r.setdefault("driver_win", 0.08)
        r.setdefault("driver_place", 0.30)
        r.setdefault("trainer_win", 0.08)
        r.setdefault("trainer_place", 0.30)
        r.setdefault("recent_runs", [])
        for key in ("career", "m12", "course", "distance", "course_distance",
                    "aw", "turf", "first_up", "second_up", "third_up"):
            r.setdefault(f"{key}_wins", 0)
            r.setdefault(f"{key}_places", 0)
            r.setdefault(f"{key}_starts", 0)

    parsed_tabs = {r["tab"] for r in runners}
    gaps = sorted(set(range(1, max(parsed_tabs) + 1)) - parsed_tabs)
    if gaps:
        warnings.append(
            "Tab number(s) " + ", ".join(str(g) for g in gaps)
            + " are missing from the parsed field - the field table may not have been read in full.")

    active = [r for r in runners if not r.get("scratched")]
    if not active:
        warnings.append("Every runner in this field is marked scratched.")
    missing = [r for r in active if r["tab"] not in blocks]
    if missing:
        warnings.append(
            f"Detailed form was not parsed for {len(missing)} active runner(s): "
            + ", ".join(f"#{r['tab']} {r['horse']}" for r in missing))
    thin = [r for r in active if len(r.get("recent_runs", [])) < 3]
    if thin:
        warnings.append(
            f"{len(thin)} active runner(s) have fewer than three completed runs on the page "
            "(barrier trials are excluded); model confidence is reduced.")
    no_ohr = [r for r in active if not any(x.get("ohr") for x in r.get("recent_runs", []))]
    if no_ohr and len(no_ohr) == len(active):
        warnings.append(
            "No official handicap ratings (OHR) on this page; the rating term falls back "
            "to its default for every runner and carries no signal.")
    return header, runners, warnings
