"""Parser for multi-meeting Bet365 race-card text exports.

The parser converts one large text export containing several meetings and races
into a canonical pre-race schema.  It preserves Bet365's analyst overview as a
separate source model while extracting independent factual form features from
runner histories.  Historical and current prices are not returned as predictive
features.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

MEETING_HEADER_RE = re.compile(
    r"^(?P<meeting>[A-Za-z][A-Za-z0-9 '&.()\-/]+?)\s{2,}(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})\s*$"
)
RUNNER_START_RE = re.compile(r"^(?P<number>\d+)\t(?P<form>[^\t]*)\t(?P<horse>.+?)\s*$")
RACE_META_RE = re.compile(r"^(?P<title>.*?)\s+(?P<distance>\d{3,5})mTrack:\s*(?P<rest>.+)$", re.I)
AGE_WEIGHT_RE = re.compile(
    r"(?P<age>\d+)YO\s+(?P<colour>[A-Z/ ]+?)\s+(?P<sex>GELDING|MARE|FILLY|COLT|HORSE)\s+(?P<weight>\d+(?:\.\d+)?)KG",
    re.I,
)
BARRIER_RE = re.compile(r"\t(?P<barrier>\d+)\s*$")
DATE_IN_RUN_RE = re.compile(r"\b(?:on\s+)?(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2})\b", re.I)
FINISH_RE = re.compile(r"\b(?P<finish>\d+)(?:st|nd|rd|th)\s+of\s+(?P<field>\d+)\b", re.I)
RUN_LAST_RE = re.compile(r"\bran\s+last\s+of\s+(?P<field>\d+)\b", re.I)
RUN_SECOND_LAST_RE = re.compile(r"\bran\s+second\s+last\s+of\s+(?P<field>\d+)\b", re.I)
SMALL_FIELD_SECOND_RE = re.compile(r"\bfinished\s+second\s+in\s+a\s+small\s+field\b", re.I)
WORD_FINISH_RE = re.compile(r"\b(?P<finish>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s+of\s+(?P<field>\d+)\b", re.I)
LAST_OF_RE = re.compile(r"\b(?:finished\s+)?last\s+of\s+(?P<field>\d+)\b", re.I)
SECOND_LAST_OF_RE = re.compile(r"\b(?:finished\s+)?second\s+last\s+of\s+(?P<field>\d+)\b", re.I)
DISTANCE_RE = re.compile(r"\bover\s+(?P<distance>\d{3,5})m\b", re.I)
WEIGHT_RE = re.compile(r"\b(?:carrying|with)\s+(?P<weight>\d+(?:\.\d+)?)kg\b", re.I)
BEHIND_RE = re.compile(r"(?P<margin>\d+(?:\.\d+)?)\s*(?:len|lengths?)\s+behind", re.I)
WIN_MARGIN_RE = re.compile(r"\bwon\s+by\s+(?P<margin>\d+(?:\.\d+)?)\s*(?:len|lengths?)\b", re.I)
WORD_MARGIN_RE = re.compile(r"\b(?P<margin>long-neck|short-neck|neck|head|nose|short head)\s+behind", re.I)

MONTHS = {name.lower(): idx for idx, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], start=1
)}

POSITIVE_SOURCE_PHRASES = {
    "clearly the one to beat": 2.6, "the one to beat": 2.3, "looks the one to beat": 2.3,
    "very hard to beat": 2.1, "hard to beat": 1.8, "leading hope": 1.7,
    "strong chance": 1.6, "strong contender": 1.6, "huge chance": 1.7,
    "top chance": 1.7, "key chance": 1.5, "winning chance": 1.5,
    "major player": 1.4, "major role": 1.3, "right in the finish": 1.4,
    "right in the mix": 1.2, "should be right there": 1.2, "go close": 1.2,
    "can win": 1.3, "can bounce back": 1.1, "can make amends": 1.1,
    "looks well placed": 1.2, "well placed": 0.9, "profiles nicely": 1.1,
    "profiles well": 1.1, "racing well": 0.9, "form hard to fault": 1.3,
    "impressive": 1.1, "dominant": 1.2, "progressing well": 1.0,
    "capable of": 0.8, "must for any top 4": 0.8, "top-three prospect": 0.9,
    "solid each way chance": 0.9, "each way chance": 0.8, "minor mix": 0.4,
    "expected to prove tough": 1.2, "looks a great chance": 1.5,
}
NEGATIVE_SOURCE_PHRASES = {
    "too tough": -1.7, "find this too tough": -1.8, "needs to improve": -1.2,
    "need to improve": -1.2, "hard to recommend": -1.6, "looks tested": -1.5,
    "not expecting": -1.2, "rough hope": -1.1, "minor claims": -0.8,
    "would need": -1.0, "doubt he can win": -1.8, "doubt she can win": -1.8,
    "has a bit to find": -1.0, "likely to find this too tough": -1.6,
    "not one of the leading": -1.3, "prepared to risk": -1.4,
    "safely held": -0.9, "below expectations": -0.8, "below his best": -0.7,
    "below her best": -0.7, "disappointed": -0.6, "can't recommend": -1.5,
    "cannot recommend": -1.5, "outside chance": -0.4,
}

FACTUAL_POSITIVE_PHRASES = {
    "raced on speed": 0.55, "tracked the speed": 0.50, "led throughout": 0.65,
    "rolled along in front": 0.55, "made ground": 0.55, "made nice improvement": 0.65,
    "ran on": 0.55, "finished on late": 0.45, "hit the line strongly": 0.65,
    "flew home": 0.70, "stayed on": 0.50, "boxed on": 0.45, "kept on": 0.40,
    "won by": 0.75, "dominant victory": 0.90, "bolting in": 0.90,
    "trial winner": 0.55, "won a recent trial": 0.60, "easy trial winner": 0.65,
    "good trial form": 0.45, "peaking": 0.45, "cherry ripe": 0.55,
    "third-up": 0.25, "second-up": 0.15, "fitter": 0.30,
}
FACTUAL_NEGATIVE_PHRASES = {
    "weakened": -0.60, "faded": -0.55, "never got into": -0.55,
    "never threatened": -0.55, "never showed": -0.70, "no impression": -0.60,
    "well beaten": -0.55, "beaten a long way": -0.60, "ran last": -0.65,
    "second last": -0.50, "eased back early": -0.15, "got too far back": -0.25,
    "not striding": -0.75, "nasal discharge": -0.85, "failed": -0.35,
    "pulled up": -0.80, "fell": -0.80, "unseated": -0.75,
}

NARRATIVE_START_RE = re.compile(
    r"^(?:At the latest run|At the latest outing|Most recently|Last start|Last outing|Last run|The latest run|"
    r"Third-up|Second-up|First-up|Resuming|Returns|Having|Comes off|Useful|Impressive|Racing well|Recent form|"
    r"Gelding by|Colt by|Filly by|Mare by|First starter|Debutant|On debut|Freshened|Back from|Yet to|Four-year-old|"
    r"Three-year-old|Two-year-old|Five-year-old|Six-year-old|Seven-year-old|Eight-year-old|Nine-year-old|Ten-year-old)",
    re.I,
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalise_name(value: str) -> str:
    text = _clean(value).upper().replace("’", "'")
    return re.sub(r"[^A-Z0-9']+", " ", text).strip()


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso_date(value: str) -> str:
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(_clean(value), fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def _days_between(race_date: str, month: str, day: int) -> Optional[int]:
    try:
        race = datetime.fromisoformat(race_date).date()
        year = race.year
        run = datetime(year, MONTHS[month.lower()], int(day)).date()
        if run > race:
            run = datetime(year - 1, MONTHS[month.lower()], int(day)).date()
        return max(0, (race - run).days)
    except Exception:
        return None


def _margin_value(text: str) -> Optional[float]:
    match = BEHIND_RE.search(text)
    if match:
        return float(match.group("margin"))
    match = WORD_MARGIN_RE.search(text)
    if match:
        return {"nose": 0.03, "short head": 0.06, "head": 0.10, "short-neck": 0.18, "neck": 0.25, "long-neck": 0.35}.get(match.group("margin").lower(), 0.25)
    return None


def _class_score(text: str) -> float:
    value = _clean(text).upper()
    if re.search(r"\b(?:GROUP\s*1|G1)\b", value):
        return 100.0
    if re.search(r"\b(?:GROUP\s*2|G2)\b", value):
        return 95.0
    if re.search(r"\b(?:GROUP\s*3|G3)\b", value):
        return 91.0
    if "LISTED" in value or re.search(r"\bLR\b", value):
        return 87.0
    if any(token in value for token in ("OPEN", "QUALITY", "QLTY", "SWP")):
        return 82.0
    bm = re.search(r"\bBM\s*(\d{2,3})\b", value)
    if bm:
        rating = float(bm.group(1))
        return max(45.0, min(82.0, 42.0 + 0.32 * rating))
    if "BENCHMARK" in value:
        bm2 = re.search(r"BENCHMARK\s*(\d{2,3})", value)
        if bm2:
            rating = float(bm2.group(1))
            return max(45.0, min(82.0, 42.0 + 0.32 * rating))
    if re.search(r"\bCLASS\s*4\b|\bCL4\b", value): return 62.0
    if re.search(r"\bCLASS\s*3\b|\bCL3\b", value): return 59.0
    if re.search(r"\bCLASS\s*2\b|\bCL2\b", value): return 55.0
    if re.search(r"\bCLASS\s*1\b|\bCL1\b", value): return 51.0
    if "MAIDEN" in value or re.search(r"\bMDN\b", value): return 43.0
    if "HANDICAP" in value or "HCP" in value: return 58.0
    return 56.0


def _going_bucket(text: str) -> str:
    value = _clean(text).upper().replace("-", " ")
    if any(token in value for token in ("HEAVY", "H(8)", "H(9)", "H(10)")):
        return "Heavy"
    if any(token in value for token in ("SOFT", "SLOW", "RAIN AFFECTED", "WET", "S(5)", "S(6)", "S(7)")):
        return "Soft"
    if any(token in value for token in ("GOOD", "G(3)", "G(4)", "FIRM")):
        return "Good"
    if any(token in value for token in ("SYNTHETIC", "ALL WEATHER", "AW")):
        return "Synthetic"
    return "Unknown"


def _sentiment(text: str, positive: Dict[str, float], negative: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
    low = _clean(text).lower()
    score = 0.0
    pos_hits: List[str] = []
    neg_hits: List[str] = []
    for phrase, value in positive.items():
        if phrase in low:
            score += value
            pos_hits.append(phrase)
    for phrase, value in negative.items():
        if phrase in low:
            score += value
            neg_hits.append(phrase)
    return score, pos_hits, neg_hits


def _form_values(form: str) -> List[int]:
    values: List[int] = []
    for char in str(form or "").lower():
        if char.isdigit():
            values.append(10 if char == "0" else int(char))
        elif char in ("f", "p", "u"):
            values.append(10)
    return values


def _form_score(form: str) -> Tuple[float, float, int]:
    values = _form_values(form)
    if not values:
        return 0.50, 0.0, 0
    quality = [max(0.0, 1.0 - (v - 1.0) / 9.0) for v in values]
    weights = list(range(1, len(quality) + 1))
    score = sum(w * q for w, q in zip(weights, quality)) / sum(weights)
    trend = quality[-1] - quality[0] if len(quality) > 1 else 0.0
    current = str(form or "").lower().split("x")[-1]
    prep_runs = sum(ch.isdigit() or ch in "fpu" for ch in current)
    return score, trend, prep_runs


def _extract_run_sentences(narrative: str, race_date: str) -> List[Dict[str, Any]]:
    # Sentence splitting is intentionally permissive because source paragraphs
    # often join clauses with semicolons.
    chunks = re.split(r"(?<=[.!?])\s+", narrative)
    runs: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = _clean(chunk)
        if not text:
            continue
        finish: Optional[int] = None
        field: Optional[int] = None
        won = False
        m = FINISH_RE.search(text)
        if m:
            finish, field = int(m.group("finish")), int(m.group("field"))
        else:
            word = WORD_FINISH_RE.search(text)
            if word:
                names = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,"eighteenth":18,"nineteenth":19,"twentieth":20}
                finish, field = names[word.group("finish").lower()], int(word.group("field"))
            else:
                m = SECOND_LAST_OF_RE.search(text) or RUN_SECOND_LAST_RE.search(text)
                if m:
                    field = int(m.group("field")); finish = max(1, field - 1)
                else:
                    m = LAST_OF_RE.search(text) or RUN_LAST_RE.search(text)
                    if m:
                        field = int(m.group("field")); finish = field
                    elif SMALL_FIELD_SECOND_RE.search(text):
                        finish = 2
        wm = WIN_MARGIN_RE.search(text)
        if wm or re.search(r"\b(?:won|winner|scored)\b", text, re.I):
            if wm or re.search(r"\bwon\s+by\b|\bscored\s+(?:a\s+)?(?:big\s+)?[\d.]+\s*(?:len|length)|\bwas too good\b", text, re.I):
                finish = 1
                won = True
        distance_match = DISTANCE_RE.search(text)
        if finish is None and not distance_match:
            continue
        date_match = DATE_IN_RUN_RE.search(text)
        days_ago = _days_between(race_date, date_match.group("month"), int(date_match.group("day"))) if date_match else None
        margin = 0.0 if won else _margin_value(text)
        if won and wm:
            margin = -float(wm.group("margin"))
        weight_match = WEIGHT_RE.search(text)
        going = _going_bucket(text)
        run = {
            "finish": finish,
            "field_size": field,
            "completed": finish is not None and finish < 98,
            "won": won or finish == 1,
            "margin": margin,
            "distance_m": int(distance_match.group("distance")) if distance_match else None,
            "days_ago": days_ago,
            "going": going,
            "weight": float(weight_match.group("weight")) if weight_match else None,
            "class_score": _class_score(text),
            "raw_text": text,
        }
        runs.append(run)
    # Deduplicate repeated clauses describing the same run.
    unique: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for run in runs:
        key = (run.get("finish"), run.get("field_size"), run.get("distance_m"), run.get("days_ago"))
        if key in seen:
            continue
        seen.add(key); unique.append(run)
    return unique[:4]


def _overview_segments(overview: str) -> Tuple[List[int], Dict[int, str]]:
    pattern = re.compile(r"(?P<name>[A-Z][A-Z0-9'’ .&\-/]+?)\s*\((?P<number>\d+)\)")
    matches = list(pattern.finditer(overview))
    order: List[int] = []
    segments: Dict[int, str] = {}
    for idx, match in enumerate(matches):
        number = int(match.group("number"))
        if number not in order:
            order.append(number)
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(overview)
        segments[number] = overview[match.start():end]
    return order, segments


def _discipline(title: str) -> str:
    text = title.upper()
    if "STEEPLE" in text or "CHASE" in text:
        return "Steeplechase"
    if "HURDLE" in text or "HDLE" in text:
        return "Hurdle"
    return "Flat"


def _looks_like_silks(value: str) -> bool:
    low = _clean(value).lower()
    colours = sum(word in low for word in ("blue", "red", "white", "green", "yellow", "black", "pink", "cerise", "purple", "orange", "gold", "navy", "maroon", "turquoise", "grey"))
    apparel = any(word in low for word in ("sleeves", "cap", "checks", "hoops", "sash", "armbands", "diamonds", "seams", "crossed"))
    return colours >= 2 or (colours >= 1 and apparel)


def _find_meeting_headers(lines: Sequence[str]) -> List[Tuple[int, re.Match[str]]]:
    found: List[Tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = MEETING_HEADER_RE.match(line.strip())
        if not match:
            continue
        # A genuine race block header is followed by TodayTomorrow and GOING.
        following = "\n".join(lines[index + 1:index + 5])
        if "TodayTomorrow" in following and "GOING:" in following:
            found.append((index, match))
    return found


class Bet365TextParser:
    """Parse one Bet365 text export containing one or many meetings."""

    def parse_file(self, path: Path | str) -> Dict[str, Any]:
        source = Path(path)
        return self.parse_text(source.read_text(encoding="utf-8", errors="replace"), source_name=source.name)

    def parse_text(self, text: str, source_name: str = "pasted_text") -> Dict[str, Any]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.splitlines()
        headers = _find_meeting_headers(lines)
        meetings: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        if not headers:
            return {"source_name": source_name, "source_hash": sha256(text.encode()).hexdigest(), "meetings": [], "warnings": ["No Bet365 race headers were detected."]}
        race_counters: Dict[str, int] = {}
        for hidx, (line_index, match) in enumerate(headers):
            end_line = headers[hidx + 1][0] if hidx + 1 < len(headers) else len(lines)
            block_lines = lines[line_index:end_line]
            meeting_name = _clean(match.group("meeting"))
            meeting_key = _normalise_name(meeting_name)
            race_counters[meeting_key] = race_counters.get(meeting_key, 0) + 1
            race_no = race_counters[meeting_key]
            race = self._parse_race_block(block_lines, meeting_name, _iso_date(match.group("date")), race_no, source_name)
            meetings.setdefault(meeting_key, {
                "meeting": meeting_name,
                "date": race.get("date", ""),
                "source_name": source_name,
                "races": [],
                "warnings": [],
            })["races"].append(race)
        meeting_list = list(meetings.values())
        for meeting in meeting_list:
            meeting["meeting_id"] = sha256(f"{meeting['date']}|{meeting['meeting']}".encode()).hexdigest()[:16]
            meeting["race_count"] = len(meeting["races"])
            meeting["active_runners"] = sum(r.get("active_field_size", 0) for r in meeting["races"])
            meeting["warnings"] = [w for r in meeting["races"] for w in r.get("warnings", [])]
        return {
            "source_name": source_name,
            "source_hash": sha256(text.encode("utf-8")).hexdigest(),
            "meetings": meeting_list,
            "warnings": warnings,
            "race_count": sum(len(m["races"]) for m in meeting_list),
        }

    def _parse_race_block(self, lines: Sequence[str], meeting: str, race_date: str, race_no: int, source_name: str) -> Dict[str, Any]:
        clean_lines = [line.rstrip() for line in lines]
        going_raw = ""
        meta_index: Optional[int] = None
        meta_match: Optional[re.Match[str]] = None
        for idx, line in enumerate(clean_lines):
            if line.strip().startswith("GOING:"):
                going_raw = _clean(line.split("GOING:", 1)[1])
            match = RACE_META_RE.match(line.strip())
            if match:
                meta_index = idx; meta_match = match; break
        warnings: List[str] = []
        if meta_match is None or meta_index is None:
            return {
                "race_no": race_no, "meeting": meeting, "date": race_date, "title": f"Race {race_no}",
                "distance_m": None, "runners": [], "warnings": ["Race metadata line was not parsed."],
                "active_field_size": 0, "declared_field_size": 0,
            }
        title = _clean(meta_match.group("title"))
        distance_m = int(meta_match.group("distance"))
        rest = meta_match.group("rest")
        rail = _clean(rest.split("Rail:", 1)[1]) if "Rail:" in rest else ""
        pre_rail = rest.split("Rail:", 1)[0]
        fields = [p.strip() for p in re.split(r"\s{2,}", pre_rail) if p.strip()]
        track_condition = fields[0] if fields else ""
        weather = fields[1] if len(fields) > 1 and not fields[1].startswith("(Max") else ""
        runner_positions = [(idx, RUNNER_START_RE.match(line)) for idx, line in enumerate(clean_lines) if RUNNER_START_RE.match(line)]
        if not runner_positions:
            warnings.append("No runner rows were detected.")
        first_runner_index = runner_positions[0][0] if runner_positions else len(clean_lines)
        overview = _clean(" ".join(clean_lines[meta_index + 1:first_runner_index]))
        source_order, overview_segments = _overview_segments(overview)
        runners: List[Dict[str, Any]] = []
        for ridx, (start_idx, start_match) in enumerate(runner_positions):
            end_idx = runner_positions[ridx + 1][0] if ridx + 1 < len(runner_positions) else len(clean_lines)
            runner = self._parse_runner_block(clean_lines[start_idx:end_idx], start_match, race_date, distance_m, track_condition)
            runners.append(runner)
        # Match source segments and suggested play after runner names are known.
        suggested_text = ""
        suggested_match = re.search(r"SUGGESTED PLAY:\s*(.+?)(?:\.|$)", overview, re.I)
        if suggested_match:
            suggested_text = _clean(suggested_match.group(1))
        for runner in runners:
            number = int(runner["number"])
            segment = overview_segments.get(number, "")
            score, pos, neg = _sentiment(segment, POSITIVE_SOURCE_PHRASES, NEGATIVE_SOURCE_PHRASES)
            runner["overview_segment"] = segment
            runner["overview_sentiment"] = score
            runner["overview_positive_phrases"] = pos
            runner["overview_negative_phrases"] = neg
            runner["source_mention_rank"] = source_order.index(number) + 1 if number in source_order else None
            normal_name = _normalise_name(runner["horse"])
            normal_suggested = _normalise_name(suggested_text)
            runner["suggested_pick"] = bool(normal_name and normal_name in normal_suggested)
            # Also match explicit runner number in rare suggested-play formats.
            if re.search(rf"\b{number}\b", suggested_text) and len(re.findall(r"\d+", suggested_text)) <= 4:
                runner["suggested_pick"] = True
        pace_sentence = overview.split(".", 1)[0] if overview else ""
        for runner in runners:
            runner["pace_mentioned"] = _normalise_name(runner["horse"]) in _normalise_name(pace_sentence)
        active = [r for r in runners if r.get("status") == "ACTIVE"]
        race_id = sha256(f"{race_date}|{meeting}|{race_no}|{title}|{distance_m}".encode()).hexdigest()[:20]
        return {
            "race_id": race_id,
            "source_name": source_name,
            "meeting": meeting,
            "date": race_date,
            "race_no": race_no,
            "title": title,
            "distance_m": distance_m,
            "track_condition": track_condition,
            "going_raw": going_raw,
            "going": _going_bucket(track_condition + " " + going_raw),
            "weather": weather,
            "rail": rail,
            "discipline": _discipline(title),
            "class_score": _class_score(title + " " + overview[:240]),
            "overview": overview,
            "source_order": source_order,
            "suggested_play": suggested_text,
            "pace_sentence": pace_sentence,
            "runners": runners,
            "declared_field_size": len(runners),
            "active_field_size": len(active),
            "warnings": warnings,
        }

    def _parse_runner_block(self, lines: Sequence[str], start_match: re.Match[str], race_date: str, race_distance: int, race_condition: str) -> Dict[str, Any]:
        lines = [line.rstrip() for line in lines if line.strip() and "© 2001-2026" not in line]
        number = int(start_match.group("number"))
        form = _clean(start_match.group("form"))
        horse = _clean(start_match.group("horse"))
        trainer = _clean(lines[1]) if len(lines) > 1 else ""
        age_line = _clean(lines[2]) if len(lines) > 2 else ""
        age_match = AGE_WEIGHT_RE.search(age_line)
        age = int(age_match.group("age")) if age_match else None
        sex = age_match.group("sex").title() if age_match else ""
        weight = float(age_match.group("weight")) if age_match else None
        barrier_idx: Optional[int] = None
        barrier: Optional[int] = None
        for idx in range(len(lines) - 1, 2, -1):
            match = BARRIER_RE.search(lines[idx])
            if match:
                barrier_idx = idx; barrier = int(match.group("barrier")); break
        jockey = ""
        if barrier_idx is not None and barrier_idx + 1 < len(lines):
            candidate = _clean(lines[barrier_idx + 1].split("\t", 1)[0])
            jockey = "" if _looks_like_silks(candidate) else candidate
        narrative_start = 3
        upper_limit = barrier_idx if barrier_idx is not None else len(lines)
        for idx in range(3, upper_limit + 1):
            if NARRATIVE_START_RE.match(_clean(lines[idx])):
                narrative_start = idx; break
        narrative_lines = list(lines[narrative_start: upper_limit + 1])
        if narrative_lines and barrier is not None:
            narrative_lines[-1] = BARRIER_RE.sub("", narrative_lines[-1]).rstrip()
        narrative = _clean(" ".join(narrative_lines))
        status = "SCRATCHED" if "SCRATCHED" in narrative.upper() else "ACTIVE"
        form_score, form_trend, prep_runs = _form_score(form)
        runs = _extract_run_sentences(narrative, race_date)
        source_score, source_pos, source_neg = _sentiment(narrative, POSITIVE_SOURCE_PHRASES, NEGATIVE_SOURCE_PHRASES)
        factual_score, factual_pos, factual_neg = _sentiment(narrative, FACTUAL_POSITIVE_PHRASES, FACTUAL_NEGATIVE_PHRASES)
        first_start = bool(re.search(r"\b(?:first starter|facing the starter for the first time|makes? (?:his|her) debut|debutant)\b", narrative, re.I)) or not form
        fitness_stage = ""
        stage_match = re.search(r"\b(First-up|Second-up|Third-up|Fourth-up|Resuming|Returns from|Back from a spell)\b", narrative, re.I)
        if stage_match:
            fitness_stage = stage_match.group(1)
        class_text_adjustment = 0.0
        low = narrative.lower()
        if any(p in low for p in ("drops in class", "down in class", "down in grade", "back in grade", "far easier", "easier race", "winnable assignment")):
            class_text_adjustment += 1.0
        if any(p in low for p in ("steps up in grade", "rise in class", "harder now", "tougher now", "much tougher")):
            class_text_adjustment -= 1.0
        going_positive = 1.0 if any(p in low for p in ("wet track to suit", "soft track to suit", "rain affected going suits", "loves this wet", "conditions suit", "track to suit")) else 0.0
        going_negative = 1.0 if any(p in low for p in ("wet track a query", "soft track a query", "doesn't handle the wet", "conditions against")) else 0.0
        on_pace = 1.0 if any(p in low for p in ("raced on speed", "tracked the speed", "led throughout", "rolled along in front", "sat on the speed", "on the pace", "go forward", "raced on the pace")) else 0.0
        closer = 1.0 if any(p in low for p in ("made ground", "ran on", "flew home", "hit the line strongly", "finished on late", "from midfield")) else 0.0
        return {
            "number": number,
            "horse": horse,
            "form": form,
            "trainer": trainer,
            "jockey": jockey,
            "age": age,
            "sex": sex,
            "weight": weight,
            "barrier": barrier,
            "status": status,
            "narrative": narrative,
            "historical_runs": runs,
            "form_score": form_score,
            "form_trend": form_trend,
            "current_prep_runs": prep_runs,
            "first_start": first_start,
            "fitness_stage": fitness_stage,
            "runner_sentiment": source_score,
            "runner_positive_phrases": source_pos,
            "runner_negative_phrases": source_neg,
            "factual_comment_score": factual_score,
            "factual_positive_phrases": factual_pos,
            "factual_negative_phrases": factual_neg,
            "class_text_adjustment": class_text_adjustment,
            "going_positive": going_positive,
            "going_negative": going_negative,
            "on_pace_history": on_pace,
            "closer_history": closer,
            "race_distance_m": race_distance,
            "race_going": _going_bucket(race_condition),
        }


def sanitize_for_export(payload: Any) -> Any:
    """Return a deep, price-free copy suitable for display and training."""
    if isinstance(payload, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in {"odds", "price", "starting_price", "current_odds", "open_odds"}:
                continue
            cleaned[key] = sanitize_for_export(value)
        return cleaned
    if isinstance(payload, list):
        return [sanitize_for_export(v) for v in payload]
    if isinstance(payload, str):
        # Remove historical dollar prices from raw narrative exports without
        # changing the parser's factual wording.
        return re.sub(r"\s+at\s+\$\d+(?:\.\d+)?", "", payload, flags=re.I)
    return payload

# Compatibility helpers used by the prediction engine and desktop interface.
def normalise_meeting_name(name: str) -> str:
    key = _normalise_name(name)
    aliases = {
        "SANDOWN LAKESIDE": "Sandown-Lakeside",
        "SUNSHINE COAST": "Sunshine Coast",
        "WYONG": "Wyong",
        "FORBES": "Forbes",
        "PORT HEDLAND": "Port Hedland",
    }
    return aliases.get(key, " ".join(part.capitalize() for part in key.lower().split()))


def parse_bet365_text(text: str, source_name: str = "pasted_text") -> List[Dict[str, Any]]:
    parsed = Bet365TextParser().parse_text(text, source_name=source_name)
    races: List[Dict[str, Any]] = []
    for meeting in parsed.get("meetings", []):
        canonical = normalise_meeting_name(meeting.get("meeting", ""))
        for race in meeting.get("races", []):
            race["meeting"] = canonical
            race["track"] = race.get("track_condition", "")
            race["going"] = race.get("going", race.get("going_raw", ""))
            race["race_class"] = race.get("class_score", 0.0)
            race["declared_runners"] = race.get("declared_field_size", len(race.get("runners", [])))
            race["active_runners"] = race.get("active_field_size", sum(1 for r in race.get("runners", []) if r.get("status") == "ACTIVE"))
            race["pace"] = {"text": race.get("pace_sentence", ""), "pace": "UNKNOWN"}
            for runner in race.get("runners", []):
                runner["name"] = runner.get("horse", "")
                runner["comment_no_odds"] = sanitize_for_export(runner.get("narrative", ""))
                runner["comment"] = runner.get("narrative", "")
                runner["form_info"] = {
                    "raw": runner.get("form", ""),
                    "finishes": _form_values(runner.get("form", "")),
                    "recent_finishes": _form_values(runner.get("form", ""))[-5:],
                    "weighted_quality": runner.get("form_score", 0.0),
                    "wins_recent": sum(1 for x in _form_values(runner.get("form", ""))[-5:] if x == 1),
                    "places_recent": sum(1 for x in _form_values(runner.get("form", ""))[-5:] if x <= 3),
                    "latest_finish": (_form_values(runner.get("form", ""))[-1] if _form_values(runner.get("form", "")) else None),
                    "trend": runner.get("form_trend", 0.0),
                }
                runs = runner.get("historical_runs", [])
                latest = runs[0] if runs else {}
                runner["latest_finish"] = latest.get("finish") or runner["form_info"].get("latest_finish")
                margin = latest.get("margin")
                if latest.get("won") and margin is not None:
                    margin = -float(margin)
                runner["latest_margin"] = margin
                runner["recent_distances"] = [x.get("distance_m") for x in runs if x.get("distance_m")][:4]
                if runner["recent_distances"]:
                    nearest = min(abs(float(x) - float(race.get("distance_m") or 0)) for x in runner["recent_distances"])
                    runner["distance_match"] = math.exp(-nearest / max(350.0, float(race.get("distance_m") or 0) * 0.25))
                else:
                    runner["distance_match"] = None
                runner["positive_comment"] = max(0.0, float(runner.get("factual_comment_score", 0.0))) + max(0.0, float(runner.get("runner_sentiment", 0.0)))
                runner["negative_comment"] = max(0.0, -float(runner.get("factual_comment_score", 0.0))) + max(0.0, -float(runner.get("runner_sentiment", 0.0)))
                runner["fitness_signal"] = 0.25 if runner.get("fitness_stage") in {"Third-up", "Second-up"} else 0.0
                runner["class_signal"] = runner.get("class_text_adjustment", 0.0)
                runner["going_signal"] = float(runner.get("going_positive", 0.0)) - float(runner.get("going_negative", 0.0))
                runner["trial_signal"] = 0.55 if any("trial" in p.lower() for p in runner.get("factual_positive_phrases", [])) else 0.0
                runner["leader_signal"] = runner.get("on_pace_history", 0.0)
                runner["closer_signal"] = runner.get("closer_history", 0.0)
                runner["jumps_experience"] = sum(0.25 for run in runs if any(k in (run.get("sentence", "").lower()) for k in ("hurdle", "chase", "steeple", "jumps")))
                runner["editorial_rank"] = runner.get("source_mention_rank")
                runner["editorial_coverage"] = 1.0 if runner.get("source_mention_rank") else 0.0
                runner["editorial_positive"] = max(0.0, float(runner.get("overview_sentiment", 0.0)))
                runner["editorial_negative"] = max(0.0, -float(runner.get("overview_sentiment", 0.0)))
                runner["suggested_play"] = bool(runner.get("suggested_pick"))
            races.append(race)
    return races


def sanitize_race_for_model(race: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_for_export(race)


def meeting_summary(races: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for race in races:
        grouped.setdefault(str(race.get("meeting", "")), []).append(race)
    return [{
        "meeting": meeting,
        "date": items[0].get("date", "") if items else "",
        "races": len(items),
        "declared_runners": sum(int(r.get("declared_runners", 0)) for r in items),
        "active_runners": sum(int(r.get("active_runners", 0)) for r in items),
        "warnings": sum(len(r.get("warnings", [])) for r in items),
    } for meeting, items in grouped.items()]
