"""Transparent harness-racing probability model for R&S Enhanced Form data."""
from __future__ import annotations

import math
from typing import Any
import numpy as np

MARKET_ALPHA = 0.56
COMPONENT_WEIGHTS = {
    "speed": 0.30,
    "tactics": 0.18,
    "trackdist": 0.12,
    "form": 0.11,
    "connections": 0.10,
    "rating": 0.08,
    "sectionals": 0.05,
    "reliability": 0.04,
    "freshness": 0.02,
}


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x=float(v); return x if math.isfinite(x) else default
    except Exception: return default


def _z(v: np.ndarray) -> np.ndarray:
    a=np.asarray(v,dtype=float); finite=np.isfinite(a)
    if not finite.any(): return np.zeros_like(a)
    med=float(np.nanmedian(a[finite])); a=np.where(finite,a,med); sd=float(a.std())
    return (a-a.mean())/(sd if sd>1e-9 else 1.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float); x=x-np.max(x); e=np.exp(np.clip(x,-30,30)); return e/max(e.sum(),1e-12)


def _market(runners: list[dict[str,Any]]) -> np.ndarray:
    inv=np.array([1/max(_safe(r.get("tab_odds"),999),1.01) for r in runners],dtype=float)
    return inv/max(inv.sum(),1e-12)


def _record_rate(r: dict[str,Any], prefix: str) -> float:
    s=max(int(r.get(f"{prefix}_starts",0)),0); w=int(r.get(f"{prefix}_wins",0)); p=int(r.get(f"{prefix}_places",0))
    win=(w+.12*8)/(s+8); top3=(w+p+.36*8)/(s+8)
    return win+.35*top3


def _weighted_recent(runs: list[dict[str,Any]], key: str, default: float, *, lower_better=False, target_dist=0, target_track="") -> float:
    vals=[]; ws=[]
    for k,run in enumerate(runs[:6]):
        if key not in run: continue
        v=_safe(run.get(key),default); w=math.exp(-.31*k)
        if target_dist and abs(int(run.get("distance",0))-target_dist)<=120: w*=1.22
        if target_track and str(run.get("track","")).upper()==target_track.upper(): w*=1.15
        vals.append(v); ws.append(w)
    if not vals: return default
    avg=float(np.average(vals,weights=ws)); return -avg if lower_better else avg


def _form_score(runs: list[dict[str,Any]]) -> float:
    vals=[]; ws=[]
    for k,run in enumerate(runs[:6]):
        if "finish" not in run: continue
        field=max(int(run.get("field",10)),2); fin=max(int(run.get("finish",field)),1)
        pct=1-(fin-1)/(field-1); margin=min(_safe(run.get("margin"),10),40)
        vals.append(pct-margin/110); ws.append(math.exp(-.31*k))
    return float(np.average(vals,weights=ws)) if vals else .35


def _connections(r: dict[str,Any]) -> float:
    dw=_safe(r.get("driver_win"),.10); dp=_safe(r.get("driver_place"),.32)
    tw=_safe(r.get("trainer_win"),.10); tp=_safe(r.get("trainer_place"),.32)
    dhn=int(r.get("driver_horse_n",0)); dtn=int(r.get("driver_trainer_n",0))
    dh=(_safe(r.get("driver_horse_win"),0)*.70+_safe(r.get("driver_horse_place"),0)*.30) if dhn else 0
    dt=(_safe(r.get("driver_trainer_win"),0)*.70+_safe(r.get("driver_trainer_place"),0)*.30) if dtn else 0
    base=.32*(.70*dw+.30*dp)+.32*(.70*tw+.30*tp)
    if dhn: base += .16*dh*min(dhn/15,1)
    if dtn: base += .20*dt*min(dtn/40,1)
    return base


def _tactics(r: dict[str,Any]) -> tuple[float,float,float]:
    runs=r.get("recent_runs",[]); vals=[]; ws=[]; lead=0; back=0
    for k,run in enumerate(runs[:6]):
        field=max(int(run.get("field",10)),2)
        positions=[run.get(x) for x in ("settle_pos","pos1200","pos800","bell_pos") if run.get(x)]
        if positions:
            pos=float(np.mean(positions[:2]))
            pace=1-(pos-1)/(field-1); vals.append(pace); ws.append(math.exp(-.30*k))
            if min(positions)<=2: lead+=1
            if max(positions)>=field-2: back+=1
    base=float(np.average(vals,weights=ws)) if vals else .45
    gate=int(r.get("gate",0)); gate_bonus=0.0
    if gate:
        gate_bonus=max(0.0,(7-gate))*0.012 if gate<=7 else -(gate-7)*0.012
    return base+gate_bonus, float(lead), float(back)


def _sectional_score(runs: list[dict[str,Any]]) -> float:
    l4=_weighted_recent(runs,"l400",28.5,lower_better=True)
    l8=_weighted_recent(runs,"l800",56.5,lower_better=True)
    return .65*l4+.35*(l8/2.0)


def _reliability(runs: list[dict[str,Any]]) -> tuple[float,int,int]:
    good=0; bad=0
    bad_terms=("gallop","broke gait","broke", "raced roughly", "unbalanced", "over race", "over raced", "checked")
    good_terms=("not fully tested","unable to secure clear running","held up")
    for run in runs[:6]:
        note=str(run.get("stewards","")).lower()
        if any(t in note for t in bad_terms): bad+=1
        if any(t in note for t in good_terms): good+=1
    return .50+.08*good-.13*bad,good,bad


def _freshness(d: float) -> float:
    d=max(d,0)
    if d<=4:return .72
    if d<=14:return 1.0-abs(d-8)/35
    if d<=35:return .80-(d-14)/90
    if d<=70:return .55-(d-35)/120
    return .20


def _components(runners: list[dict[str,Any]], header: dict[str,Any]) -> dict[str,Any]:
    dist=int(header.get("distance_m",0) or 0); track=str(header.get("track","")).upper()
    speed=[]; tactics=[]; td=[]; form=[]; conn=[]; rating=[]; sect=[]; reli=[]; fresh=[]; tactical_meta=[]; reliability_meta=[]
    for r in runners:
        runs=r.get("recent_runs",[])
        adj=_weighted_recent(runs,"mile_rate_adj",0.0,lower_better=True,target_dist=dist,target_track=track)
        imr=_weighted_recent(runs,"imr",115.0,lower_better=True,target_dist=dist,target_track=track)
        speed.append(.72*adj+.28*(imr/8.0))
        tac,lead,back=_tactics(r); tactics.append(tac); tactical_meta.append((lead,back))
        td.append(.48*_record_rate(r,"course_distance")+.32*_record_rate(r,"distance")+.20*_record_rate(r,"course"))
        form.append(_form_score(runs)); conn.append(_connections(r))
        rating.append(_weighted_recent(runs,"ohr",62.0,lower_better=False,target_dist=dist,target_track=track))
        sect.append(_sectional_score(runs))
        rel,g,b=_reliability(runs); reli.append(rel); reliability_meta.append((g,b))
        fresh.append(_freshness(_safe(r.get("dls"),14)))
    adj_good=[]; imr_good=[]
    for r in runners:
        runs=r.get("recent_runs",[])
        adj_good.append(_weighted_recent(runs,"mile_rate_adj",0.0,lower_better=True,target_dist=dist,target_track=track))
        imr_good.append(_weighted_recent(runs,"imr",115.0,lower_better=True,target_dist=dist,target_track=track))
    speed_z=.72*_z(np.array(adj_good,dtype=float))+.28*_z(np.array(imr_good,dtype=float))
    comp={
        "speed":speed_z,"tactics":_z(np.array(tactics)),"trackdist":_z(np.array(td)),"form":_z(np.array(form)),
        "connections":_z(np.array(conn)),"rating":_z(np.array(rating)),"sectionals":_z(np.array(sect)),
        "reliability":_z(np.array(reli)),"freshness":_z(np.array(fresh)),
        "speed_raw":np.array(speed),"tactics_raw":np.array(tactics),"trackdist_raw":np.array(td),"form_raw":np.array(form),
        "connections_raw":np.array(conn),"rating_raw":np.array(rating),"sectionals_raw":np.array(sect),
        "reliability_raw":np.array(reli),"freshness_raw":np.array(fresh),"tactical_meta":tactical_meta,"reliability_meta":reliability_meta,
    }
    pace=.82*comp["tactics"]+.18*(-_z(np.array([r.get("gate",8) for r in runners],dtype=float)))
    order=np.argsort(-pace); adj=np.zeros(len(runners))
    if len(order)>1:
        gap=pace[order[0]]-pace[order[1]]
        if gap>.65: adj[order[0]]+=.18
        elif gap<.20: adj[order[0]]-=.04; adj[order[1]]-=.04
    comp["pace"]=pace; comp["pace_adjust"]=adj; comp["early_order"]=order
    return comp


def _simulate(p: np.ndarray, sims: int, seed: int):
    n=len(p); rng=np.random.default_rng(seed); counts=np.zeros((n,n),dtype=int)
    for _ in range(sims):
        rem=list(range(n))
        for pos in range(n):
            w=p[rem]; w=w/w.sum(); j=int(rng.choice(len(rem),p=w)); idx=rem.pop(j); counts[idx,pos]+=1
    pp=counts/sims; return pp[:,:min(2,n)].sum(1),pp[:,:min(3,n)].sum(1),(pp*np.arange(1,n+1)).sum(1)


def predict(all_runners: list[dict[str,Any]], header: dict[str,Any], *, alpha: float=MARKET_ALPHA, sims: int=20000, seed: int=42) -> dict[str,Any]:
    active=[r for r in all_runners if not r.get("scratched")]
    if len(active)<2: raise ValueError("At least two active harness runners are required.")
    p_mkt=_market(active); comp=_components(active,header); score=np.zeros(len(active))
    for k,w in COMPONENT_WEIGHTS.items(): score+=w*comp[k]
    score+=comp["pace_adjust"]
    p_fund=_softmax(1.10*score); a=float(np.clip(alpha,0,1))
    p_win=_softmax(a*np.log(np.clip(p_mkt,1e-9,1))+(1-a)*np.log(np.clip(p_fund,1e-9,1)))
    top2,top3,exp_pos=_simulate(p_win,max(int(sims),1000),int(seed)); order=list(np.argsort(-p_win))
    early_order=list(comp["early_order"]); early_rank=np.empty(len(active),dtype=int)
    for rank,idx in enumerate(early_order,1): early_rank[idx]=rank
    odds=np.array([_safe(r.get("tab_odds"),999) for r in active]); ev=p_win*odds-1; fair=1/np.clip(p_win,1e-9,None)
    conf=[]
    for i,r in enumerate(active):
        completeness=min(len(r.get("recent_runs",[]))/5,1)+(.18 if r.get("course_distance_starts",0) else 0)+(.12 if r.get("driver_l50_n",0) else 0)
        completeness=min(completeness/1.30,1); vals=np.array([comp[k][i] for k in COMPONENT_WEIGHTS]); agreement=1/(1+float(vals.std()))
        rk=order.index(i); sep=p_win[i]-p_win[order[rk+1]] if rk+1<len(order) else p_win[i]
        conf.append(int(np.clip(round(2+3*completeness+2*agreement+5*max(sep,0)),1,9)))
    recs=[]; why=[]
    for i,r in enumerate(active):
        rk=order.index(i)+1
        rec="TOP PICK" if rk==1 else "DANGER" if rk<=3 else "VALUE" if ev[i]>=.12 and p_win[i]>=.06 else "EXOTICS / PLACE" if top3[i]>=.34 else "WATCH"
        recs.append(rec); lead,back=comp["tactical_meta"][i]; good,bad=comp["reliability_meta"][i]
        reasons=[f"Projected tactical position {early_rank[i]} of {len(active)} from gate {r.get('gate','?')} and recent in-running positions.",
                 f"Market {p_mkt[i]*100:.1f}% vs fundamental {p_fund[i]*100:.1f}%; blended win {p_win[i]*100:.1f}%.",
                 f"Driver {r.get('driver','')} L50 {r.get('driver_win',0)*100:.0f}% win; trainer {r.get('trainer','')} L50 {r.get('trainer_win',0)*100:.0f}% win.",
                 f"Recent tactical flags: {int(lead)} forward/leader runs, {int(back)} backmarker runs; steward reliability flags {int(good)} positive / {int(bad)} negative.",
                 f"Fair odds ${fair[i]:.2f} vs listed ${odds[i]:.2f}; EV {ev[i]:+.2f}."]
        why.append(" ".join(reasons))
    gap=p_win[order[0]]-p_win[order[1]] if len(order)>1 else p_win[order[0]]
    overall=int(np.clip(round(4+15*gap+np.mean(conf)/3),1,9))
    return {"runners":active,"p_mkt":p_mkt,"p_fund":p_fund,"p_win":p_win,"top2":top2,"top3":top3,"exp_pos":exp_pos,"fair":fair,"ev_win":ev,
            "order":order,"early_order":early_order,"early_rank":early_rank,"components":comp,"conf":conf,"recs":recs,"why":why,"overall_conf":overall}
