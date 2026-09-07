"""Parser for Racing & Sports thoroughbred Enhanced Form pages.

Design note
-----------
The field table is read via its **header row**, never by column position.  R&S
ship several column sets for the same Enhanced Form page and freely leave cells
empty (a horse with no declared jockey, a meeting with no Bet365 column), so any
positional scheme silently mis-assigns columns the moment a blank appears.  Rows
keep their empty cells when split, and a row needs only a tab number and a horse
name to produce a runner -- everything else is best-effort.  A gap in the tab
sequence raises a warning rather than quietly shrinking the field.
"""
from __future__ import annotations

import re
from typing import Any

# Bookmaker column headers and inline price prefixes seen on R&S pages.
BOOKMAKERS = (
    "betfair", "bet365", "tab", "tabtouch", "ladbrokes", "sportsbet",
    "neds", "unibet", "pointsbet", "bluebet", "boombet", "palmerbet",
)
_NAME_HEADERS = ("horse", "runner", "greyhound")
_PRICE_CELL = re.compile(r"^\$?(\d+(?:\.\d+)?)$")
# "8%-24%-50" -> win%, place%, sample
_PCT_TRIPLE = re.compile(r"(\d+(?:\.\d+)?)%-(\d+(?:\.\d+)?)%-(\d+)")
# "0-1-12" -> wins, places, starts
_WPS = re.compile(r"(\d+)-(\d+)-(\d+)")


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


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .-\t")


def _clean_md(text: str) -> str:
    """Normalise clipboard text: strip markdown pipes/links, unify newlines."""
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)          # [label](url)
    t = re.sub(r"[ \t]*\|[ \t]*", "\t", t) if "|--" in t else t
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
        r"(?mi)^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(.+?\d{4})$", t
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
    # "1400m TURF SOFT 5" / "1200m ALL WEATHER GOOD" / "1000m DIRT HEAVY 10"
    m = re.search(
        r"(?mi)^(\d{3,4})m\s+([A-Z][A-Z ]*?)\s+(FIRM|GOOD|SOFT|HEAVY|SYNTHETIC|SLOW|FAST|WET)\s*(\d+)?\s*$",
        t,
    )
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
    """Locate columns of a `Tab | Horse | WT | BP | Jockey | ... ` header row."""
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
        "wt": find("wt", "weight"),
        "bp": find("bp", "barrier", "draw"),
        "jockey": find("jockey", "rider"),
        "jrat": find("jrat", "j rat"),
        "trainer": trainer,
        "trat": find("trat", "t rat"),
        # Bookmaker columns to the right of the trainer, in page order.
        "prices": [i for i, c in enumerate(low) if i > anchor and c in BOOKMAKERS],
        "anchor": anchor,
    }


def _rating(cell: str) -> float:
    """`H 2.8` / `3.2` / `-` -> float. The H prefix flags a highlighted rating."""
    return _f(re.sub(r"^\s*H\s*", "", str(cell or ""), flags=re.I), 0.0)


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

        # Price: right-most bookmaker column carrying a plain number. Using the
        # header to find these columns keeps the TRat rating from being read as
        # a price when every bookmaker cell happens to be blank.
        odds, book = 999.0, ""
        idxs = colmap["prices"] or list(range(colmap["anchor"] + 1, len(cells)))
        for i in reversed(idxs):
            m = _PRICE_CELL.match(cell(i))
            if m:
                odds = _f(m.group(1), 999.0)
                book = "tab"
                break

        jockey_raw = cell(colmap["jockey"])
        claim = re.search(r"\(a(\d+(?:\.\d+)?)\)", jockey_raw, re.I)
        runner: dict[str, Any] = {
            "tab": tab,
            "horse": name,
            "wt": _f(cell(colmap["wt"])),
            "bp": _i(re.sub(r"[^\d]", "", cell(colmap["bp"])) or 0),
            "jockey": _clean_name(re.sub(r"\s*\(a[\d.]+\)", "", jockey_raw, flags=re.I)),
            "claim": _f(claim.group(1)) if claim else 0.0,
            "jrat": _rating(cell(colmap["jrat"])),
            "trainer": _clean_name(cell(colmap["trainer"])),
            "trat": _rating(cell(colmap["trat"])),
            "scratched": scratch,
            "tab_odds": 999.0 if scratch else odds,
            "price_book": "" if scratch else book,
        }
        out.append(runner)
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


# Form figures line followed by a bookmaker price: "58x74" / "betfair$28".
_FORM_PRICE = re.compile(
    r"(?mi)^\s*([0-9xX]{1,10})\s*$\s*^\s*([A-Za-z][A-Za-z0-9]*)?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)


def _runner_blocks(raw: str, runners: list[dict[str, Any]]) -> dict[int, str]:
    """Split the page into per-horse blocks anchored on `NAME  <n>yo`."""
    t = _clean_md(raw)
    found: list[tuple[int, int]] = []
    for r in runners:
        m = _horse_header_re(r["horse"]).search(t)
        if m:
            found.append((m.start(), r["tab"]))
    found.sort()
    blocks: dict[int, str] = {}
    for idx, (pos, tab) in enumerate(found):
        end = found[idx + 1][0] if idx + 1 < len(found) else len(t)
        # Reach back for the form-figures/price pair that precedes the header.
        prefix_start = max(0, pos - 260)
        pairs = list(_FORM_PRICE.finditer(t[prefix_start:pos]))
        start = prefix_start + pairs[-1].start() if pairs else pos
        blocks[tab] = t[start:end]
    return blocks


def _glued_panel(seg: str, labels: list[str]) -> dict[str, str]:
    """Read an R&S glued panel, where each line is `<value><next label>`.

        Car
        0-1-1212m      -> Car = "0-1-12", next label "12m"
        0-1-9Crs       -> 12m = "0-1-9",  next label "Crs"

    Splitting on a regex alone is ambiguous because some labels start with a
    digit ("12m"), so `0-1-1212m` would greedily read starts as 1212. Walking
    the panel in order and peeling the *known* next label off each line's end
    removes the ambiguity entirely.
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
        suffix = next(
            (lb for lb in known if ln.endswith(lb) and len(ln) > len(lb)), None
        )
        if suffix is None:
            if ln in labels:            # bare label line, no value in between
                current = ln
                continue
            out[current] = ln           # final value in the panel
            current = None
            continue
        out[current] = ln[: -len(suffix)].strip()
        current = suffix
    return out


_FILTER_LABELS = [
    "Car", "12m", "Crs", "Dist", "Crs & Dist", "Firm", "Good", "Soft",
    "Heavy", "AW", "Turf", "G1", "G2", "G3", "LR", "FU", "2U", "3U",
    "ClockW", "AClockW", "Dirt", "Sand",
]
_FACT_LABELS = [
    "Car PM", "12m PM", "RTC/km", "RunsDistTC", "DLS", "DLW", "DOD",
    "ROI", "For-Against", "Hdle", "Stpl",
]


def _parse_filters(body: str) -> dict[str, tuple[int, int, int]]:
    """Parse the WPS Filters panel into {label: (wins, places, starts)}."""
    m = re.search(r"(?is)\bFilters.*?(?=\nFacts\b|\nDays Since Last Run|\nBest Winning|\Z)", body)
    if not m:
        return {}
    raw = _glued_panel(m.group(0), _FILTER_LABELS)
    out: dict[str, tuple[int, int, int]] = {}
    for label, value in raw.items():
        g = _WPS.fullmatch(value.strip())
        if g:
            out[label] = (int(g.group(1)), int(g.group(2)), int(g.group(3)))
    return out


def _parse_facts(body: str) -> dict[str, str]:
    """Parse the Facts panel (same glued layout) into {label: raw value}."""
    m = re.search(r"(?is)\nFacts\b.*?(?=\nDays Since Last Run|\nBest Winning|\Z)", body)
    if not m:
        return {}
    return _glued_panel(m.group(0), _FACT_LABELS)


def _segment(body, start_pat, end_pats):
    """Text between a line-anchored start label and the first of several stops."""
    m = re.search(start_pat, body, re.M | re.I)
    if not m:
        return ""
    rest = body[m.end():]
    stops = [e.start() for e in
             (re.search(pat, rest, re.M | re.I) for pat in end_pats) if e]
    return rest[:min(stops)] if stops else rest


def _pct_in(seg, label):
    """`Last508%-24%-50` inside a segment -> (0.08, 0.24, 50).

    Anchored on the owning label rather than counting how many times `Last50`
    appears on the page: a horse with no declared jockey has an *empty* Last50
    line, and counting occurrences would hand the trainer's strike rate to the
    jockey.
    """
    m = re.search(re.escape(label) + r"\s*" + _PCT_TRIPLE.pattern, seg, re.I)
    if not m:
        return 0.0, 0.0, 0
    return _f(m.group(1)) / 100.0, _f(m.group(2)) / 100.0, int(m.group(3))


def _parse_recent_runs(block: str) -> list[dict[str, Any]]:
    """Parse completed race lines. Barrier trials and spells are skipped."""
    runs: list[dict[str, Any]] = []
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        fin = re.fullmatch(r"\s*(\d{1,2}) of (\d{1,2})\s*", ln)
        if not fin:
            continue
        # The matching Results line is the next one starting with a date.
        res = ""
        for j in range(i + 1, min(i + 14, len(lines))):
            if re.match(r"\s*\d{1,2} \w{3} \d{4} \(", lines[j]):
                res = lines[j]
                break
        if not res:
            continue
        run: dict[str, Any] = {
            "finish": int(fin.group(1)),
            "field_size": int(fin.group(2)),
        }
        # Lines between "N of M" and the Results line carry the run-up tag
        # ("2U - 22d"), the official handicap rating ("59" then "OHR"), the
        # placegetters and the race/sectional times.
        for k in range(i + 1, min(i + 14, len(lines))):
            ln_k = lines[k].strip()
            if ln_k == res.strip():
                break
            m_tag = re.fullmatch(r"(FU|FRS|\d+U)\s*-\s*(.+)", ln_k)
            if m_tag:
                run["runup_tag"] = m_tag.group(1).upper()
                continue
            if re.fullmatch(r"\d{2,3}", ln_k) and k + 1 < len(lines) and lines[k + 1].strip() == "OHR":
                run["ohr"] = int(ln_k)
                continue
            m_t = re.match(r"R\.Time\s+(\d+):(\d+(?:\.\d+)?)\s+Sec\.Time\s+([\d.]+)", ln_k)
            if m_t:
                run["race_time_s"] = int(m_t.group(1)) * 60 + float(m_t.group(2))
                sec = _f(m_t.group(3))
                run["sec600_s"] = sec if sec > 20 else None
                continue

        def grab(pat: str, cast=_f, default=None):
            m = re.search(pat, res, re.I)
            return cast(m.group(1)) if m else default

        run["date"] = grab(r"^\s*(\d{1,2} \w{3} \d{4})", str, "")
        run["days_ago"] = grab(r"\((\d+)d ago\)", _i, None)
        run["track"] = grab(r"\)\s+([A-Z][A-Za-z' ]+?)\s+\(", str, "").strip()
        run["margin"] = grab(r"Margin\s+([\d.]+)L")
        run["distance"] = grab(r"Distance\s+(\d+)m", _i)
        run["surface"] = grab(r"Surface\s+(\w+)", str, "")
        run["going"] = grab(r"SOT\s+([A-Z])", str, "")
        run["going_rating"] = grab(r"SOT\s+[A-Z]\s+(\d+)", _i, None)
        run["race_class"] = grab(r"Class\s+(.+?)\s+Prize", str, "")
        run["api"] = grab(r"\bAPI\s+([\d.]+)")
        run["sp"] = grab(r"\bSP\s+\$([\d.]+)")
        run["weight"] = grab(r"\bWeight\s+([\d.]+)")
        run["cd"] = grab(r"\bCD\s+([\d.]+)")
        run["bp"] = grab(r"\bBP\s+(\d+)", _i)
        run["jockey"] = _clean_name(grab(r"Jockey\s+([A-Z][A-Za-z' .-]+?)\s+Weight", str, ""))
        run["settle_pos"] = grab(r"(\d+)(?:st|nd|rd|th) Place on settling", _i)
        run["turn_pos"] = grab(r"(\d+)(?:st|nd|rd|th) Place on turn", _i)
        run["tempo"] = grab(r"Tempo\s+(.+?)(?:\s+Video Comments|\s+Race/Horse|$)", str, "")
        m = re.search(r"Stewards\s+(.+?)(?:\s+Inrunning|\s+Video Comments|$)", res, re.I)
        run["stewards"] = m.group(1).strip() if m else ""
        run["prize"] = grab(r"Prize AUD \$([\d,]+)")
        run["direction"] = grab(r"Track Direction\s+(Anti-Clockwise|Clockwise)", str, "")
        run["pos_800"] = grab(r"(\d+)(?:st|nd|rd|th) Place at 800m", _i)
        m = re.search(r"Video Comments\s+(.+)$", res, re.I)
        run["comment"] = m.group(1).strip() if m else ""
        low = (run["stewards"] + " " + run["comment"]).lower()
        run["slow_begin"] = bool(re.search(r"slow (to begin|out|away)|began (very )?awkwardly|failed to muster", low))
        runs.append(run)
    return runs


def _parse_block(block: str, runner: dict[str, Any]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    hm = _horse_header_re(runner["horse"]).search(block)
    head = block[: hm.start()] if hm else block[:300]
    body = block[hm.start():] if hm else block

    # form figures + market price
    pair = _FORM_PRICE.search(head) or _FORM_PRICE.search(block[:400])
    if pair:
        d["form"] = pair.group(1).lower()
        book = (pair.group(2) or "").lower()
        price = _f(pair.group(3), 999.0)
        d["price_source"] = book or "unknown"
        if book == "betfair":
            d["bf_odds"] = price
        elif book in ("tab", "tabtouch"):
            # A TAB price, not an exchange price. Only fill in if the field
            # table carried none; `parse` mirrors it into bf_odds afterwards.
            if _f(runner.get("tab_odds"), 999.0) >= 999.0:
                d["tab_odds"] = price
            d["book_odds"] = price
        else:
            d["book_odds"] = price          # Ladbrokes, Bet365, ...

    # age / colour / sex / declared weight from the block header line
    if hm:
        m = re.match(
            r"(?i)^\s*.+?\s+(\d+)yo\s+([A-Z/ ]+?)\s+(Gelding|Mare|Colt|Filly|Horse|Stallion|Rig)"
            r"(?:\s*\(BP:\s*(\d+)\))?(?:\s+([\d.]+)kg)?",
            hm.group(0),
        )
        if m:
            d["age"] = _i(m.group(1))
            d["colour"] = m.group(2).strip()
            d["sex"] = m.group(3)
            if m.group(4):
                d["bp_block"] = _i(m.group(4))
            if m.group(5):
                d["wt_block"] = _f(m.group(5))

    for key, label in (("sire", "Sire"), ("dam", "Dam"), ("dam_sire", "Dam Sire")):
        m = re.search(rf"(?mi)^{label}([A-Z][^\n]*)$", body)
        if m:
            d[key] = _clean_name(m.group(1))

    # Jockey/Trainer strike rates. Last50 appears twice (jockey then trainer).
    jockey_seg = _segment(body, r"^Jockey", [r"^Trainer", r"^Raced Dist\."])
    trainer_seg = _segment(body, r"^Trainer", [r"^Raced Dist\.", r"^J/H", r"^Win Dist\."])
    d["jky_win"], d["jky_place"], d["jky_n"] = _pct_in(jockey_seg, "Last50")
    d["trn_win"], d["trn_place"], d["trn_n"] = _pct_in(trainer_seg, "Last50")
    d["jh_win"], d["jh_place"], d["jh_n"] = _pct_in(body, "J/H")
    d["jt_win"], d["jt_place"], d["jt_n"] = _pct_in(body, "J/T")

    m = re.search(r"(?i)Raced Dist\.\s*(\d+)m\s*-\s*(\d+)m", body)
    if m:
        d["dist_min"], d["dist_max"] = _i(m.group(1)), _i(m.group(2))
    m = re.search(r"(?mi)^Win Dist\.\s*([^\n]*)$", body)
    if m:
        d["win_dists"] = [int(x) for x in re.findall(r"(\d{3,4})m", m.group(1))]

    m = re.search(r"(?i)Days Since Last Run:\s*(\d+)\s*days(?:\s*\((FU|\d+U)\))?", body)
    if m:
        d["dslr"] = _i(m.group(1))
        # R&S tag the run of the preparation: (FU) first-up, (2U) second-up, ...
        if m.group(2):
            tag = m.group(2).upper()
            d["runup_tag"] = tag
            d["runup"] = 1 if tag == "FU" else _i(tag.rstrip("U"))

    filt = _parse_filters(body)
    d["filters"] = filt
    for label, key in (
        ("Car", "Car"), ("12m", "M12"), ("Crs", "Crs"), ("Dist", "Dist"),
        ("Crs & Dist", "CrsDist"), ("Firm", "Firm"), ("Good", "Good"),
        ("Soft", "Soft"), ("Heavy", "Heavy"), ("AW", "AW"), ("Turf", "Turf"),
        ("FU", "FU"), ("2U", "U2"), ("3U", "U3"),
    ):
        w, p, s = filt.get(label, (0, 0, 0))
        d[f"{key}_wins"], d[f"{key}_places"], d[f"{key}_starts"] = w, p, s
        d[f"{key}_win"] = (w / s) if s else 0.0
        d[f"{key}_plc"] = ((w + p) / s) if s else 0.0

    facts = _parse_facts(body)
    d["facts"] = facts
    if "DLW" in facts:
        d["dlw"] = facts["DLW"]
    d["maiden"] = str(facts.get("DLW", "")).lower().startswith("maiden")

    runs = _parse_recent_runs(body)
    d["recent_runs"] = runs
    if runs:
        d["last_fin"] = runs[0]["finish"]
        d["last_margin"] = runs[0].get("margin")
        d["last_sp"] = runs[0].get("sp")
        apis = [r["api"] for r in runs if r.get("api")]
        if apis:
            d["api_avg"] = sum(apis) / len(apis)
        ohrs = [r["ohr"] for r in runs if r.get("ohr")]
        if ohrs:
            d["ohr"] = ohrs[0]            # most recent official rating
            d["ohr_max"] = max(ohrs)
    return d


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
def parse(raw: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    header = parse_header(raw)
    runners = parse_summary(raw)
    if not runners:
        return header, [], ["Could not locate the runners field table."]

    blocks = _runner_blocks(raw, runners)
    for r in runners:
        if r["tab"] in blocks:
            r.update(_parse_block(blocks[r["tab"]], r))
        if r.get("scratched"):
            r["tab_odds"] = 999.0
            r["bf_odds"] = 999.0
        # Prefer a real exchange price; fall back to any bookmaker, then TAB.
        r.setdefault("bf_odds", r.get("book_odds") or r.get("tab_odds", 999.0))
        r.setdefault("form", "")
        r.setdefault("dslr", 30)
        r.setdefault("last_fin", 10)
        r.setdefault("ohr", 0)
        r.setdefault("jky_win", 0.05)
        r.setdefault("trn_win", 0.05)
        r.setdefault("recent_runs", [])
        for key in ("Car", "M12", "Crs", "Dist", "CrsDist", "Firm", "Good",
                    "Soft", "Heavy", "AW", "Turf", "FU", "U2", "U3"):
            r.setdefault(f"{key}_wins", 0)
            r.setdefault(f"{key}_places", 0)
            r.setdefault(f"{key}_starts", 0)
            r.setdefault(f"{key}_win", 0.0)
            r.setdefault(f"{key}_plc", 0.0)

    parsed_tabs = {r["tab"] for r in runners}
    gaps = sorted(set(range(1, max(parsed_tabs) + 1)) - parsed_tabs)
    if gaps:
        warnings.append(
            "Tab number(s) " + ", ".join(str(g) for g in gaps)
            + " are missing from the parsed field - the field table may not have been read in full."
        )

    active = [r for r in runners if not r.get("scratched")]
    if not active:
        warnings.append("Every runner in this field is marked scratched.")
    missing_detail = [r for r in active if r["tab"] not in blocks]
    if missing_detail:
        warnings.append(
            f"Detailed form was not parsed for {len(missing_detail)} active runner(s): "
            + ", ".join(f"#{r['tab']} {r['horse']}" for r in missing_detail)
        )
    no_jockey = [r for r in active if not r.get("jockey")]
    if no_jockey:
        warnings.append(
            f"{len(no_jockey)} active runner(s) have no declared jockey; "
            "jockey strike-rate terms fall back to defaults."
        )
    thin = [r for r in active if len(r.get("recent_runs", [])) < 3]
    if thin:
        warnings.append(
            f"{len(thin)} active runner(s) have fewer than three completed runs on the page; "
            "model confidence is reduced."
        )
    if all(_f(r.get("ohr"), 0) <= 0 for r in active):
        warnings.append(
            "No official ratings on this page (normal for a maiden). The official-rating "
            "term contributes nothing and the fundamental model leans on the other features."
        )
    return header, runners, warnings


# --------------------------------------------------------------------------
# Speed Map page (AES / AFS pace values)
# --------------------------------------------------------------------------
_SM_ROW = re.compile(r"(?m)^\s*(\d{1,2})\t+([A-Z][A-Z0-9' .\-]+?)\s*$")


def parse_speed_map(raw: str) -> dict[int, dict[str, float]]:
    """Read the R&S Speed Map "Pace Values" table into {tab: {aes, afs, bp, jr}}.

    The copied table puts each runner on three lines: ``1\t\tNAME``, the
    breeding line, then ``63.0\tJOCKEY\t4.1\t5\t17.0\t17.6``. AES/AFS are the
    last two numbers of that third line; BP and JR sit just before them. Only
    the trailing numbers are trusted, so a missing jockey cell cannot shift
    the pairing.
    """
    t = _clean_md(raw)
    out: dict[int, dict[str, float]] = {}
    rows = list(_SM_ROW.finditer(t))
    for idx, m in enumerate(rows):
        tab = int(m.group(1))
        end = rows[idx + 1].start() if idx + 1 < len(rows) else len(t)
        seg = t[m.end():end]
        for ln in seg.splitlines():
            cells = [c.strip() for c in ln.split("\t")]
            nums = [c for c in cells if _PRICE_CELL.match(c)]
            if len(nums) >= 4:
                aes, afs = _f(nums[-2]), _f(nums[-1])
                if 12.0 <= aes <= 20.0 and 12.0 <= afs <= 20.0:
                    out[tab] = {"aes": aes, "afs": afs,
                                "bp": _f(nums[-3]), "jr": _f(nums[-4])}
                    break
    return out


def track_direction(runners: list[dict[str, Any]], track: str = "") -> str:
    """Majority "Track Direction" of the field's past runs, preferring runs at
    today's track. Returns "clockwise" or "anticlockwise"."""
    votes: dict[str, int] = {}
    home: dict[str, int] = {}
    tk = (track or "").upper().split()[0] if track else ""
    for r in runners:
        for run in r.get("recent_runs", []):
            d = (run.get("direction") or "").lower()
            if not d:
                continue
            key = "anticlockwise" if d.startswith("anti") else "clockwise"
            votes[key] = votes.get(key, 0) + 1
            if tk and tk in (run.get("track") or "").upper():
                home[key] = home.get(key, 0) + 1
    pool = home or votes
    if not pool:
        return "clockwise"
    return max(pool, key=pool.get)
