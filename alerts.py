"""Kick-off alerts for the live board: imminence test, beep sound, cell colours."""
from __future__ import annotations

import io
import math
import struct
import wave
from datetime import datetime
from functools import lru_cache
from typing import Optional

import soccerway as SW

IMMINENT_SECONDS = 5 * 60

# Row colours
PURPLE = "#e9d5ff"        # blinking: kick-off within 5 minutes
ROSE = "#fbcfe8"          # started or finished
# Odds cell colours (same scheme as the Steamers board)
ODDS_GREEN = "#c8f7c5"    # under 2.0
ODDS_YELLOW = "#fff3b0"   # under 3.0
ODDS_BLUE = "#d6ecff"     # 3.0 and above


def seconds_to_kickoff(m: SW.Match, now: datetime) -> float:
    return (m.kickoff - now).total_seconds()


def imminent(m: SW.Match, now: datetime) -> bool:
    """Scheduled and kicking off within IMMINENT_SECONDS (but not yet started)."""
    if m.stage != "1":
        return False
    s = seconds_to_kickoff(m, now)
    return 0 < s <= IMMINENT_SECONDS


def started(m: SW.Match, now: datetime) -> bool:
    """Live or finished per the feed, or past its kick-off time while the feed
    still says scheduled (the feed can lag by a few minutes)."""
    return m.stage in ("2", "3") or seconds_to_kickoff(m, now) <= 0


def odds_colour(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x <= 0:
        return ""
    if x < 2.0:
        return f"background-color: {ODDS_GREEN};"
    if x < 3.0:
        return f"background-color: {ODDS_YELLOW};"
    return f"background-color: {ODDS_BLUE};"


@lru_cache(maxsize=1)
def beep_wav(freq: float = 880.0, seconds: float = 0.35, rate: int = 22050) -> bytes:
    """A short sine beep as WAV bytes (mono, 16-bit)."""
    n = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            t = i / rate
            env = min(1.0, i / (rate * 0.02), (n - i) / (rate * 0.05))   # soft attack / release
            frames += struct.pack("<h", int(32767 * 0.6 * env * math.sin(2 * math.pi * freq * t)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def new_alerts(matches: list[SW.Match], now: datetime, already: set[str]) -> list[SW.Match]:
    """Matches that just became imminent and have not been announced yet."""
    return [m for m in matches if imminent(m, now) and m.id not in already]
