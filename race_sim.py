"""Turn a rated field into smooth in-running trajectories.

The race is described at a handful of checkpoints (jump, early, settled,
mid-race, turn, straight, finish) by two numbers per horse: the gap in lengths
behind whoever leads at that point, and the lane (metres off the rail). Both
are interpolated with cubic Hermite splines over distance covered, so the
animation shows the leaders roll forward, the field settle into position, the
closers wind up from the turn and the finishing margins the model expects.

Settling positions come from the Speed Map's early-speed values plus each
horse's typical settling position; finishing margins come from the rating
engine. A "random draw" mode adds race-day noise so the same field produces a
different, plausible running each time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from race_model import LENGTH_M, SIGMA_LENGTHS


def hermite(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation with Catmull-Rom style finite-difference slopes.
    `ys` has shape (n_series, len(xs)); returns (n_series, len(xq))."""
    xs = np.asarray(xs, float)
    ys = np.atleast_2d(np.asarray(ys, float))
    m = np.zeros_like(ys)
    m[:, 1:-1] = (ys[:, 2:] - ys[:, :-2]) / (xs[2:] - xs[:-2])
    m[:, 0] = (ys[:, 1] - ys[:, 0]) / (xs[1] - xs[0])
    m[:, -1] = (ys[:, -1] - ys[:, -2]) / (xs[-1] - xs[-2])
    idx = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, len(xs) - 2)
    h = xs[idx + 1] - xs[idx]
    t = (xq - xs[idx]) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (ys[:, idx] * h00 + m[:, idx] * h * h10 + ys[:, idx + 1] * h01 + m[:, idx + 1] * h * h11)


@dataclass
class Simulation:
    distance: float
    tabs: list[int]
    names: list[str]
    checkpoints: np.ndarray          # distance covered by the leader at each checkpoint
    cp_labels: list[str]
    gap_cp: np.ndarray               # (n, k) lengths behind the leader
    lane_cp: np.ndarray              # (n, k) metres off the rail
    s_grid: np.ndarray               # dense grid of leader distance
    gap: np.ndarray                  # (n, len(s_grid))
    lane: np.ndarray                 # (n, len(s_grid))
    finish_margin: np.ndarray        # lengths behind the winner
    finish_order: list[int]          # indices, winner first
    pred_time_s: float
    mode: str
    seed: int
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.tabs)

    def state_at(self, s_leader: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """(distance covered per horse in m, lane in m, running order indices) when
        the leader has covered `s_leader` metres."""
        s_leader = float(np.clip(s_leader, 0.0, self.distance))
        j = int(np.clip(np.searchsorted(self.s_grid, s_leader), 0, len(self.s_grid) - 1))
        gaps = self.gap[:, j]
        lanes = self.lane[:, j]
        dist = s_leader - gaps * LENGTH_M
        order = [int(i) for i in np.argsort(-dist, kind="stable")]
        return dist, lanes, order

    def leader_distance(self, u: float) -> float:
        """Leader's distance covered at normalised animation time u in [0, 1]."""
        u = float(np.clip(u, 0.0, 1.0))
        return float(self.extras["s_of_u"](u))

    def race_clock(self, u: float) -> float:
        return self.pred_time_s * self.leader_distance(u) / self.distance

    def phase(self, s_leader: float) -> str:
        d = self.distance
        if s_leader < 120:
            return "The jump"
        if s_leader < 0.32 * d:
            return "Settling down"
        if s_leader < d - 650:
            return "Mid-race"
        if s_leader < d - 350:
            return "On the turn"
        if s_leader < d - 5:
            return "Home straight"
        return "Finish"

    def order_at(self, s_leader: float) -> list[tuple[int, float]]:
        dist, _, order = self.state_at(s_leader)
        lead = dist[order[0]]
        return [(int(i), float((lead - dist[i]) / LENGTH_M)) for i in order]


def _settle_gaps(es: np.ndarray, n: int) -> np.ndarray:
    """Early-speed scores -> lengths behind the leader once the field settles."""
    spread = 2.5 + 0.55 * n                      # a 15-horse field strings out ~11 L
    rng = es.max() - es.min()
    if rng < 1e-9:
        return np.linspace(0, spread, n)
    return spread * (es.max() - es) / rng


def _profile(u: float) -> float:
    """Leader speed multiplier over normalised time: slow first strides, a lull
    in the middle, then a lift turning for home."""
    return 1.0 - 0.55 * math.exp(-u / 0.03) - 0.06 * math.sin(math.pi * u) + 0.08 * max(0.0, u - 0.7) / 0.3


def simulate(rated: dict, mode: str = "expected", seed: int = 0,
             sigma: float = SIGMA_LENGTHS) -> Simulation:
    runners = rated["runners"]
    meta = rated["meta"]
    n = len(runners)
    D = float(meta["distance_m"])
    rng = np.random.default_rng(seed)

    tabs = [r["tab"] for r in runners]
    names = [r["horse"] for r in runners]
    rating = np.array([r["rating"] for r in runners])
    es = np.array([r["early_speed"] for r in runners])
    late = np.array([r["late_speed"] for r in runners])
    bp = np.array([r["bp"] for r in runners], dtype=float)
    slow = np.array([r["slow_begin"] for r in runners])

    if mode == "random":
        es = es + rng.normal(0, 0.45, n)
        final_L = -(rating + rng.normal(0, sigma, n))
        jump_jitter = rng.uniform(0, 0.8, n)
    else:
        final_L = -rating
        jump_jitter = 0.15 * ((bp * 7) % 3)          # deterministic, tiny
    final_L = final_L - final_L.min()
    if final_L.max() > 15:
        final_L *= 15 / final_L.max()

    settle = _settle_gaps(es, n)
    jump = jump_jitter + np.where(slow, 1.2, 0.0) + 0.02 * bp
    jump -= jump.min()

    # blend weights toward the final margins; closers (late > 0) hold back longer
    lam_turn = np.clip(0.45 - 0.12 * late, 0.2, 0.7)
    lam_straight = np.clip(0.82 - 0.06 * late, 0.65, 0.95)

    cps = np.array([0.0, 0.10 * D, 0.32 * D, 0.58 * D, D - 400.0, D - 150.0, D])
    labels = ["Jump", "Early", "Settled", "Mid-race", "Turn", "Straight", "Finish"]
    g = np.zeros((n, len(cps)))
    g[:, 0] = jump
    g[:, 1] = 0.45 * settle + 0.55 * jump
    g[:, 2] = settle
    g[:, 3] = 0.85 * settle + 0.15 * final_L
    g[:, 4] = (1 - lam_turn) * settle + lam_turn * final_L
    g[:, 5] = (1 - lam_straight) * settle + lam_straight * final_L
    g[:, 6] = final_L
    g -= g.min(axis=0, keepdims=True)               # the leader at each checkpoint is at 0

    # lanes (metres off the rail)
    bp_rank = np.argsort(np.argsort(bp))
    settle_rank = np.argsort(np.argsort(g[:, 2] + 1e-6 * bp))
    fin_rank = np.argsort(np.argsort(final_L + 1e-6 * bp))
    lane = np.zeros_like(g)
    lane[:, 0] = 0.6 + 0.8 * bp_rank                # barrier stalls, inside to out
    lane[:, 1] = 0.8 + 0.35 * bp_rank + 0.6 * ((settle_rank + bp_rank) % 3)
    col = (settle_rank // 2 + bp_rank) % 3
    lane[:, 2] = 0.8 + 1.3 * col
    lane[:, 3] = lane[:, 2]
    lane[:, 4] = 0.8 + 1.3 * col + 0.3 * (late > 0.5)
    fcol = (fin_rank + (bp_rank // 3)) % 4
    lane[:, 5] = 0.8 + 1.4 * fcol
    lane[:, 6] = lane[:, 5]
    if mode == "random":
        lane[:, 2:] += rng.uniform(-0.25, 0.25, (n, 5))

    s_grid = np.linspace(0.0, D, int(D) + 1)
    gap = hermite(cps, g, s_grid)
    gap -= gap.min(axis=0, keepdims=True)
    lane_d = np.clip(hermite(cps, lane, s_grid), 0.5, 20.0)

    finish_order = list(np.argsort(final_L, kind="stable"))

    # leader distance as a function of normalised time
    us = np.linspace(0, 1, 1201)
    v = np.array([_profile(u) for u in us])
    s = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(us))])
    s = s / s[-1] * D

    def s_of_u(u: float) -> float:
        return float(np.interp(u, us, s))

    return Simulation(
        distance=D, tabs=tabs, names=names, checkpoints=cps, cp_labels=labels,
        gap_cp=g, lane_cp=lane, s_grid=s_grid, gap=gap, lane=lane_d,
        finish_margin=final_L, finish_order=[int(i) for i in finish_order],
        pred_time_s=float(meta.get("pred_time_s") or D / 16.6), mode=mode, seed=seed,
        extras={"s_of_u": s_of_u, "early_speed": es, "late_speed": late},
    )


def running_calls(sim: Simulation) -> list[dict]:
    """A text 'race call' at each checkpoint: who leads and by how much."""
    calls = []
    for s_cp, label in zip(sim.checkpoints, sim.cp_labels):
        order = sim.order_at(float(s_cp))
        calls.append({
            "checkpoint": label,
            "leader_distance": float(s_cp),
            "to_go": float(sim.distance - s_cp),
            "order": [(sim.tabs[i], sim.names[i], gap) for i, gap in order],
        })
    return calls
