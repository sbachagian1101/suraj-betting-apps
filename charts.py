"""Plotly figures for the prediction page.  Every function takes the rated rows
(and sometimes the simulation) and returns a figure; nothing here fetches."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from model import Rated, SimResult

BUILD = "2026-09-07c"      # bumped with every change; app.py refuses a stale copy

PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#be185d",
           "#4d7c0f", "#b45309", "#1d4ed8", "#0f766e", "#9333ea", "#c2410c", "#475569",
           "#65a30d", "#db2777", "#0369a1", "#a16207", "#7e22ce", "#334155"]


def _lab(r: Rated) -> str:
    return f"{r.number} {r.name}"


def _layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def probability_chart(rows: list[Rated]) -> go.Figure:
    rows = list(rows)[::-1]
    labels = [_lab(r) for r in rows]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[r.p_final * 100 for r in rows], name="Final", orientation="h",
                marker_color="#2563eb", text=[f"{r.p_final:.0%}" for r in rows], textposition="outside")
    fig.add_bar(y=labels, x=[r.p_model * 100 for r in rows], name="Form model", orientation="h",
                marker_color="#16a34a", opacity=0.75)
    if any(r.p_market for r in rows):
        fig.add_bar(y=labels, x=[(r.p_market or 0) * 100 for r in rows], name="Market",
                    orientation="h", marker_color="#94a3b8", opacity=0.9)
    fig.update_layout(barmode="group", xaxis_title="Win probability (%)")
    return _layout(fig, height=max(360, 34 * len(rows) + 80))


def speed_map(rows: list[Rated], dist_m: int | None) -> go.Figure:
    fig = go.Figure()
    xs, ys, sizes, texts, cols = [], [], [], [], []
    for i, r in enumerate(rows):
        sp = r.speed_score if r.speed_score is not None else 0.5
        xs.append(sp * 10)
        ys.append(r.entry.barrier or 0)
        sizes.append(14 + 60 * r.p_final)
        texts.append(_lab(r))
        cols.append(PALETTE[i % len(PALETTE)])
    fig.add_scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="middle right",
                    marker=dict(size=sizes, color=cols, opacity=0.8, line=dict(width=1, color="white")),
                    hovertemplate="%{text}<br>early speed %{x:.1f}/10<br>barrier %{y}<extra></extra>")
    fig.update_xaxes(title="Early speed  (0 = backmarker, 10 = leader)", range=[-1, 12.5])
    fig.update_yaxes(title="Barrier", autorange="reversed")
    fig.add_annotation(x=10, y=0.5, text="LEAD", showarrow=False, opacity=0.5)
    fig.add_annotation(x=0, y=0.5, text="REAR", showarrow=False, opacity=0.5)
    return _layout(fig, height=460, showlegend=False,
                   title=f"Speed map{f' - {dist_m}m' if dist_m else ''}  (marker size = win chance)")


def form_heatmap(rows: list[Rated], max_runs: int = 8) -> go.Figure:
    labels = [_lab(r) for r in rows]
    z = np.full((len(rows), max_runs), np.nan)
    text = [[""] * max_runs for _ in rows]
    for i, r in enumerate(rows):
        for j, run in enumerate(r.entry.runs[:max_runs]):
            if run.non_finish:
                text[i][j] = run.comment[:3] or "NF"
                z[i, j] = 12
            elif run.margin_l is not None:
                z[i, j] = min(max(run.margin_l, -2), 12)
                text[i][j] = (f"{run.pos}/{run.field_size or '?'}<br>{run.margin_l:+.1f}L"
                              if run.pos else f"{run.margin_l:+.1f}L")
            elif run.pos:
                text[i][j] = f"{run.pos}/{run.field_size or '?'}"
                z[i, j] = min(1.4 * (run.pos - 1) ** 0.9, 12)
    hover = [[f"{labels[i]}<br>{(r.entry.runs[j].run_date or '')} {r.entry.runs[j].track} "
              f"{r.entry.runs[j].dist_m or '?'}m {r.entry.runs[j].race_class}"
              if j < len(r.entry.runs) else "" for j in range(max_runs)] for i, r in enumerate(rows)]
    fig = go.Figure(go.Heatmap(
        z=z[::-1], y=labels[::-1], x=[f"Run {j + 1}" for j in range(max_runs)],
        text=np.array(text)[::-1], texttemplate="%{text}", hovertext=np.array(hover)[::-1],
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=[[0, "#16a34a"], [0.25, "#a3e635"], [0.5, "#fde047"], [1, "#dc2626"]],
        zmin=-2, zmax=12, colorbar=dict(title="Beaten<br>lengths")))
    fig.update_xaxes(title="Latest run first")
    return _layout(fig, height=max(360, 40 * len(rows) + 80))


def odds_movement(rows: list[Rated]) -> go.Figure:
    fig = go.Figure()
    with_flucs = [r for r in rows if len(r.entry.flucs) >= 2 and all(f.when for f in r.entry.flucs)]
    if with_flucs:
        for i, r in enumerate(rows):
            fl = r.entry.flucs
            if len(fl) < 2 or not all(f.when for f in fl):
                continue
            fig.add_scatter(x=[f.when for f in fl], y=[f.price for f in fl], mode="lines+markers",
                            name=_lab(r), line=dict(color=PALETTE[i % len(PALETTE)]))
        fig.update_yaxes(title="Win price", type="log")
        fig.update_xaxes(title="Time (UTC)")
        return _layout(fig, height=440, title="Price fluctuations")
    labels = [_lab(r) for r in rows][::-1]
    opens = [r.entry.odds_open or r.odds or 0 for r in rows][::-1]
    nows = [r.odds or 0 for r in rows][::-1]
    cols = ["#dc2626" if r.move_label == "drifter" else "#16a34a" if r.move_label == "steamer"
            else "#94a3b8" for r in rows][::-1]
    fig.add_bar(y=labels, x=opens, orientation="h", name="Opening", marker_color="#cbd5e1")
    fig.add_bar(y=labels, x=nows, orientation="h", name="Now", marker_color=cols)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Win price (green = firmed, red = drifted)")
    return _layout(fig, height=max(360, 30 * len(rows) + 80), title="Opening vs current price")


def connections_chart(rows: list[Rated]) -> go.Figure:
    labels = [_lab(r) for r in rows][::-1]

    def pct(st):
        return (st[1] / st[0] * 100) if st and st[0] else None
    fig = go.Figure()
    fig.add_bar(y=labels, x=[pct(r.entry.jockey_stats) for r in rows][::-1], orientation="h",
                name="Jockey win %", marker_color="#2563eb")
    fig.add_bar(y=labels, x=[pct(r.entry.trainer_stats) for r in rows][::-1], orientation="h",
                name="Trainer win %", marker_color="#f59e0b")
    fig.add_bar(y=labels, x=[pct(r.entry.combo_stats) for r in rows][::-1], orientation="h",
                name="Combination win %", marker_color="#7c3aed")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Strike rate (%)")
    return _layout(fig, height=max(360, 30 * len(rows) + 80))


def finish_heatmap(sim: SimResult) -> go.Figure:
    n = len(sim.names)
    labels = [f"{num} {nm}" for num, nm in zip(sim.numbers, sim.names)]
    fig = go.Figure(go.Heatmap(
        z=sim.position_matrix[::-1] * 100, y=labels[::-1], x=[str(i + 1) for i in range(n)],
        colorscale="Blues", texttemplate="%{z:.0f}", colorbar=dict(title="%")))
    fig.update_xaxes(title="Finishing position", side="top")
    return _layout(fig, height=max(360, 34 * n + 80), title="Simulated finishing positions (20,000 runs)")


def terms_chart(rows: list[Rated], top_n: int = 8) -> go.Figure:
    rows = list(rows)[:top_n]
    keys = [k for k in rows[0].terms if any(abs(r.terms.get(k, 0)) > 0.02 for r in rows)] if rows else []
    fig = go.Figure()
    for i, k in enumerate(keys):
        fig.add_bar(x=[_lab(r) for r in rows], y=[r.terms.get(k, 0) for r in rows], name=k,
                    marker_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="Rating contribution (lengths)")
    return _layout(fig, height=440, title="What drives each rating")


def sensitivity_chart(tally: dict[str, float]) -> go.Figure:
    names = list(tally)[:8]
    fig = go.Figure(go.Bar(x=[tally[n] * 100 for n in names], y=names, orientation="h",
                           marker_color="#2563eb", text=[f"{tally[n]:.0%}" for n in names],
                           textposition="outside"))
    fig.update_xaxes(title="Share of jittered model runs rating this runner top (%)", range=[0, 110])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=320, title="Robustness of the top pick")
