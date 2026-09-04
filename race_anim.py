"""Render a Simulation as an animated GIF (and still snapshots) with Pillow.

Flat colours, a fixed 96-colour palette and frame-difference encoding keep a
30-second, 12 fps GIF to a few megabytes. No matplotlib: each frame is drawn
directly with ImageDraw, which is also far faster on Streamlit Cloud.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from race_model import LENGTH_M, fmt_time
from race_sim import Simulation

W, H = 960, 560
PANEL_X = 640
MAX_DURATION_S = 30
GAP_SCALE = 3.5        # display metres per real metre of gap, so the field is readable
LANE_PX = 4.2          # display pixels per metre off the rail

# Australian saddlecloth colours (fill, number colour)
SADDLECLOTH = {
    1: ("#d7191c", "#ffffff"), 2: ("#ffffff", "#000000"), 3: ("#1f3a93", "#ffffff"),
    4: ("#f9d71c", "#000000"), 5: ("#1a9641", "#ffffff"), 6: ("#111111", "#f9d71c"),
    7: ("#f57c00", "#000000"), 8: ("#f48fb1", "#000000"), 9: ("#26c6da", "#000000"),
    10: ("#6a1b9a", "#ffffff"), 11: ("#9e9e9e", "#d7191c"), 12: ("#aeea00", "#000000"),
    13: ("#795548", "#ffffff"), 14: ("#800000", "#f9d71c"), 15: ("#f9d71c", "#1f3a93"),
    16: ("#42a5f5", "#f9d71c"), 17: ("#0d1b4b", "#ffffff"), 18: ("#42a5f5", "#ffffff"),
    19: ("#1a9641", "#f9d71c"), 20: ("#42a5f5", "#f57c00"), 21: ("#26c6da", "#111111"),
    22: ("#e040fb", "#ffffff"), 23: ("#5d4037", "#f9d71c"), 24: ("#ff7043", "#ffffff"),
}
BG = "#0f1f14"
INFIELD = "#1b5e20"
TURF = "#3e9a4b"
AW = "#b98a52"
RAIL = "#f5f5f5"
PANEL = "#141a1f"
INK = "#f2f2f2"
MUTED = "#9aa5ad"
ACCENT = "#ffca28"


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                 # very old Pillow
        return ImageFont.load_default()


class Track:
    """A rounded-rectangle course, arc-length parametrised. The drawn oval is a
    fixed size; `course_m` metres of racecourse are mapped onto its circumference
    so any race distance fits the canvas."""

    def __init__(self, course_m: float = 1800.0, straight: float = 280.0, radius: float = 118.0,
                 cx: float = 322.0, cy: float = 290.0, clockwise: bool = True,
                 finish_offset_m: float = 70.0):
        self.L, self.R, self.cx, self.cy = straight, radius, cx, cy
        self.clockwise = clockwise
        self.C = 2 * straight + 2 * math.pi * radius          # circumference in px
        self.course_m = float(course_m)
        self.k = self.C / self.course_m                          # px per metre of course
        self.finish_s = finish_offset_m * self.k

    def point(self, s: float, lane_px: float = 0.0) -> tuple[float, float]:
        """Position `s` metres *before* the winning post (s=0 is the post),
        offset `lane_px` toward the outside of the track."""
        # Moving forward means s decreasing. A clockwise race advances along the
        # canonical (clockwise) parameter, so s counts *back* from the post.
        s_px = s * self.k
        p = (self._finish_p() + (-s_px if self.clockwise else s_px)) % self.C
        return self._canon(p, lane_px)

    def _finish_p(self) -> float:
        # The canonical parameter runs clockwise from the left end of the top
        # straight. The bottom straight (home straight, in front of the stands)
        # therefore runs right->left; a clockwise race finishes near its left
        # end, an anticlockwise race near its right end.
        start_bottom = self.L + math.pi * self.R
        if self.clockwise:
            return start_bottom + self.L - self.finish_s
        return start_bottom + self.finish_s

    def _canon(self, p: float, lane_px: float) -> tuple[float, float]:
        L, R, cx, cy = self.L, self.R, self.cx, self.cy
        r = R + lane_px
        if p < L:                                    # top straight, left -> right
            return cx - L / 2 + p, cy - r
        p -= L
        if p < math.pi * R:                          # right semicircle, top -> bottom
            a = -math.pi / 2 + p / R
            return cx + L / 2 + r * math.cos(a), cy + r * math.sin(a)
        p -= math.pi * R
        if p < L:                                    # bottom straight, right -> left
            return cx + L / 2 - p, cy + r
        p -= L                                       # left semicircle, bottom -> top
        a = math.pi / 2 + p / R
        return cx - L / 2 + r * math.cos(a), cy + r * math.sin(a)

    def outline(self, lane_px: float, steps: int = 240) -> list[tuple[float, float]]:
        return [self._canon(self.C * i / steps, lane_px) for i in range(steps)]


def course_length(distance: float) -> float:
    """Metres of racecourse mapped onto the oval: a full lap holds the race with room to spare."""
    return max(1800.0, distance + 400.0)


def _draw_track(draw: ImageDraw.ImageDraw, track: Track, distance: float, surface: str,
                band: float) -> None:
    turf = AW if surface.upper().startswith(("AW", "SYN", "DIRT", "SAND", "ALL")) else TURF
    draw.polygon(track.outline(band), fill=turf)
    draw.polygon(track.outline(0.0), fill=INFIELD)
    # the part of the course this race does not use is shaded
    unused = track.course_m - distance
    if unused > 30:
        pts = [track.point(distance + unused * i / 200, band * 0.5) for i in range(201)]
        draw.line(pts, fill="#2f7d3a", width=int(band) - 6)
    draw.line(track.outline(band) + track.outline(band)[:1], fill="#e0e0e0", width=2)
    draw.line(track.outline(0.0) + track.outline(0.0)[:1], fill=RAIL, width=3)
    # distance markers every 200 m
    f = _font(11)
    for m in range(200, int(distance), 200):
        a, b = track.point(m, -4), track.point(m, band + 4)
        draw.line([a, b], fill="#ffffff", width=1)
        lx, ly = track.point(m, band + 16)
        draw.text((lx, ly), f"{m}", fill="#dddddd", font=f, anchor="mm")
    # start gates and winning post
    a, b = track.point(distance, -6), track.point(distance, band + 6)
    draw.line([a, b], fill="#cfd8dc", width=6)
    gx, gy = track.point(distance, -24)              # inside the rail: never clipped by the canvas
    draw.text((gx, gy), f"START {int(distance)}m", fill="#ffffff", font=_font(11, True), anchor="mm")
    a, b = track.point(0, -10), track.point(0, band + 10)
    draw.line([a, b], fill="#ffffff", width=5)
    draw.line([a, b], fill="#d7191c", width=2)
    px, py = track.point(0, band + 26)
    draw.text((px, py), "WINNING POST", fill=ACCENT, font=_font(11, True), anchor="mm")
    # direction arrow on the far straight
    ax, ay = track.point(distance * 0.5, -22)
    bx, by = track.point(distance * 0.5 - 40 / track.k, -22)
    draw.line([(ax, ay), (bx, by)], fill="#c8e6c9", width=3)
    ang = math.atan2(by - ay, bx - ax)
    for sgn in (1, -1):
        draw.line([(bx, by), (bx - 10 * math.cos(ang + sgn * 0.5), by - 10 * math.sin(ang + sgn * 0.5))],
                  fill="#c8e6c9", width=3)


@lru_cache(maxsize=8)
def _background(distance: float, surface: str, clockwise: bool, band: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    track = Track(course_length(distance), clockwise=clockwise)
    _draw_track(draw, track, distance, surface, band)
    draw.rectangle([PANEL_X, 0, W, H], fill=PANEL)
    draw.line([PANEL_X, 0, PANEL_X, H], fill="#2b343b", width=2)
    return img


def pretty_name(name: str) -> str:
    """SHE'SADARE -> She'sadare, FREDDY'S SHOCK -> Freddy's Shock."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in str(name).split())


def _colour(tab: int) -> tuple[str, str]:
    return SADDLECLOTH.get(((tab - 1) % 24) + 1, ("#888888", "#ffffff"))


def render_frame(sim: Simulation, s_leader: float, header: dict, clockwise: bool = True,
                 clock_s: float | None = None, final: bool = False) -> Image.Image:
    D = sim.distance
    band = 58.0
    surface = str(header.get("surface") or "TURF")
    img = _background(D, surface, clockwise, band).copy()
    draw = ImageDraw.Draw(img)
    track = Track(course_length(D), clockwise=clockwise)

    dist, lanes, order = sim.state_at(s_leader)
    r = 9
    for i in reversed(order):                        # leaders drawn last, on top
        gap_m = s_leader - float(dist[i])            # real metres behind the leader
        s_back = D - s_leader + gap_m * GAP_SCALE    # exaggerated for legibility
        off = min(band - 5.0, 4.0 + float(lanes[i]) * LANE_PX)
        x, y = track.point(max(0.0, s_back), off)
        fill, ink = _colour(sim.tabs[i])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline="#111111", width=1)
        draw.text((x, y), str(sim.tabs[i]), fill=ink, font=_font(10, True), anchor="mm")

    # ---- side panel --------------------------------------------------------
    x0 = PANEL_X + 16
    right = W - 16
    title = f"{header.get('track', 'Race')} R{header.get('race_no', '')}".strip()
    draw.text((x0, 14), title, fill=INK, font=_font(20, True))
    going = f"{str(header.get('going') or '').title()} {header.get('going_rating') or ''}".strip()
    sub = f"{int(D)}m {surface.title()} {going}".strip()
    draw.text((x0, 40), sub, fill=MUTED, font=_font(13))
    if header.get("race_name"):
        draw.text((x0, 58), str(header["race_name"])[:34], fill=MUTED, font=_font(12))

    phase = "Finish" if final else sim.phase(s_leader)
    draw.text((x0, 84), phase.upper(), fill=ACCENT, font=_font(16, True))
    to_go = max(0, int(round(D - s_leader)))
    draw.text((right, 84), f"{to_go}m to go" if to_go > 0 and not final else "RESULT",
              fill=INK, font=_font(15, True), anchor="ra")
    if clock_s is not None:
        draw.text((right, 106), fmt_time(clock_s), fill=MUTED, font=_font(13), anchor="ra")
    draw.text((x0, 106), "Predicted running order", fill=MUTED, font=_font(12))
    draw.line([x0, 124, right, 124], fill="#2b343b", width=1)

    lead = dist[order[0]]
    row_h = min(27, int((H - 160) / max(1, len(order))))
    f_name = _font(13, True) if row_h >= 24 else _font(12, True)
    f_gap = _font(12)
    for k, i in enumerate(order):
        y = 134 + k * row_h
        cy = y + (row_h - 5) / 2
        fill, ink = _colour(sim.tabs[i])
        draw.rounded_rectangle([x0, y, x0 + 22, y + row_h - 5], radius=4, fill=fill, outline="#000000")
        draw.text((x0 + 11, cy), str(sim.tabs[i]), fill=ink, font=_font(10, True), anchor="mm")
        draw.text((x0 + 30, cy), f"{k + 1}", fill=MUTED, font=f_gap, anchor="lm")
        draw.text((x0 + 52, cy), pretty_name(sim.names[i])[:20], fill=INK, font=f_name, anchor="lm")
        gap_L = (lead - dist[i]) / LENGTH_M
        txt = "" if k == 0 else (f"{gap_L:.1f}L" if gap_L >= 0.15 else "hd" if gap_L >= 0.05 else "nose")
        draw.text((right, cy), txt, fill=MUTED if k else ACCENT, font=f_gap, anchor="rm")

    draw.text((x0, H - 22), "Model prediction, not the actual result. Market weight capped at 10%.",
              fill="#5f6b73", font=_font(10))
    return img


def render_gif(sim: Simulation, header: dict, clockwise: bool = True, duration_s: float = 24.0,
               fps: int = 12, hold_s: float = 2.5) -> bytes:
    duration_s = float(min(max(duration_s, 5.0), MAX_DURATION_S))
    n_frames = max(2, int(round((duration_s - hold_s) * fps)))
    frames: list[Image.Image] = []
    durations: list[int] = []
    for k in range(n_frames):
        u = k / (n_frames - 1)
        s = sim.leader_distance(u)
        frames.append(render_frame(sim, s, header, clockwise, clock_s=sim.race_clock(u),
                                   final=(k == n_frames - 1)))
        durations.append(int(round(1000 / fps)))
    durations[-1] = int(hold_s * 1000)
    base = frames[0].quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal_frames = [base] + [f.quantize(palette=base, dither=Image.Dither.NONE) for f in frames[1:]]
    buf = io.BytesIO()
    pal_frames[0].save(buf, format="GIF", save_all=True, append_images=pal_frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=1)
    return buf.getvalue()


def snapshots(sim: Simulation, header: dict, clockwise: bool = True) -> list[tuple[str, Image.Image]]:
    D = sim.distance
    pts = [("Just after the start", 150.0), ("Mid-race", 0.5 * D),
           ("Turning for home", D - 380.0), ("The finish", D)]
    us = np.linspace(0, 1, 201)
    s_of_u = np.array([sim.leader_distance(x) for x in us])
    out = []
    for label, s in pts:
        u = float(np.interp(s, s_of_u, us))
        out.append((label, render_frame(sim, s, header, clockwise, clock_s=sim.race_clock(u),
                                        final=(s >= D))))
    return out


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
