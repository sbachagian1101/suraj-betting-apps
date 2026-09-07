"""Plotly charts for a single match's analysis."""
from __future__ import annotations

import plotly.graph_objects as go

import model as M
from pipeline import Analysis

GREEN, GREY, RED, LIME, BLUE = "#2ea043", "#adb5bd", "#e4572e", "#c6ff00", "#1f77b4"


def _layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=20, r=20, t=50, b=30),
                      legend=dict(orientation="h", y=-0.2), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def model_vs_book(A: Analysis) -> go.Figure:
    p = A.pred
    labels = [A.prof_h.name, "Draw", A.prof_a.name]
    model = [p.p_home, p.p_draw, p.p_away]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Model", x=labels, y=model, marker_color=BLUE,
                         text=[f"{v:.0%}" for v in model], textposition="outside"))
    imp = M.implied(A.odds) if A.odds else None
    if imp:
        book = [imp["home"], imp["draw"], imp["away"]]
        fig.add_trace(go.Bar(name="Book (margin removed)", x=labels, y=book, marker_color=GREY,
                             text=[f"{v:.0%}" for v in book], textposition="outside"))
    fig.update_yaxes(tickformat=".0%", range=[0, max(model + ([0.0] if not imp else book)) * 1.25])
    fig.update_layout(barmode="group")
    return _layout(fig, "1X2: model v book")


def outcome_pie(A: Analysis) -> go.Figure:
    p = A.pred
    sig = M.bet_signal(p, A.odds)
    labels = [A.prof_h.name, "Draw", A.prof_a.name]
    vals = [p.p_home, p.p_draw, p.p_away]
    pull = [0, 0, 0]
    colors = [GREEN, GREY, RED]
    if sig and sig["bet"]:
        i = {"home": 0, "draw": 1, "away": 2}[sig["side"]]
        pull[i] = 0.12
        colors[i] = LIME
    fig = go.Figure(go.Pie(labels=labels, values=vals, pull=pull, hole=0.35,
                           marker=dict(colors=colors, line=dict(color="#222", width=1)),
                           texttemplate="%{label}<br>%{percent}", sort=False))
    return _layout(fig, "Win / draw / win", height=340)


def score_heatmap(A: Analysis, size: int = 6) -> go.Figure:
    g = [row[:size] for row in A.pred.grid[:size]]
    fig = go.Figure(go.Heatmap(
        z=g, x=[str(i) for i in range(size)], y=[str(i) for i in range(size)],
        colorscale=[[0, "#f6fbf6"], [1, "#1a7f37"]], showscale=False,
        text=[[f"{v:.1%}" for v in row] for row in g], texttemplate="%{text}",
        hovertemplate=f"{A.prof_h.name} %{{y}} - {A.prof_a.name} %{{x}}: %{{text}}<extra></extra>"))
    fig.update_xaxes(title=A.prof_a.name, side="top")
    fig.update_yaxes(title=A.prof_h.name, autorange="reversed")
    return _layout(fig, "Scoreline probabilities", height=380)


def xg_history(A: Analysis) -> go.Figure:
    fig = go.Figure()
    for prof, color in ((A.prof_h, GREEN), (A.prof_a, RED)):
        recs = list(reversed(prof.records))              # oldest -> newest
        x = [f"{r.match.kickoff:%d %b} {'v' if r.side == 'H' else '@'} {r.opponent}" for r in recs]
        fig.add_trace(go.Scatter(name=f"{prof.name} xG", x=x, y=[r.xg_for for r in recs],
                                 mode="lines+markers", line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(name=f"{prof.name} xGA", x=x, y=[r.xg_against for r in recs],
                                 mode="lines+markers", line=dict(color=color, width=1.5, dash="dot")))
        fig.add_trace(go.Bar(name=f"{prof.name} goals", x=x, y=[r.goals_for for r in recs],
                             marker_color=color, opacity=0.25))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="per match")
    return _layout(fig, "Recent matches: xG (solid), xGA (dotted), goals (bars)", height=380)


def rating_bars(A: Analysis) -> go.Figure | None:
    if not (A.la_h and A.la_a):
        return None
    fig = go.Figure()
    for la, name, color in ((A.la_h, A.prof_h.name, GREEN), (A.la_a, A.prof_a.name, RED)):
        rows = [r for r in la.rows if r["Avg rating"] is not None]
        fig.add_trace(go.Bar(name=name, x=[r["Player"] for r in rows], y=[r["Avg rating"] for r in rows],
                             marker_color=color))
    fig.update_yaxes(title="avg rating, last N", range=[5, 9])
    fig.update_layout(barmode="group")
    return _layout(fig, "Starting XI ratings", height=360)
