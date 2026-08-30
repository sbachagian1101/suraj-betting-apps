"""Score the engine's picks against actual finishing order.

Results are recorded as the first four tab numbers, which is what the results
strip on the Racing & Sports meeting page shows.
"""
from __future__ import annotations

import sys

import engine as E
import meeting as M

# CARNARVON (WA), 30 Aug 2026 - first four tab numbers per race, read off the
# meeting's results strip. Race 6 had not been run.
RESULTS = {
    "2026-08-30-CARNARVON-T.xlsx": {
        1: [4, 7, 2, 1],
        2: [7, 4, 2, 3],
        3: [6, 4, 1, 7],
        4: [7, 5, 2, 3],
        5: [6, 4, 3, 1],
    },
}


def run(path: str, results: dict[int, list[int]], sims: int = 20000):
    df = M.read_meeting(path)
    blocks = {b.number: b for b in M.split_races(df)}
    rows = []
    for rno, finish in sorted(results.items()):
        b = blocks.get(rno)
        if b is None:
            print(f"  R{rno}: not found in the file")
            continue
        a = E.analyse_race_text(b.text, simulations=sims)
        order = [p.tab for p in a.predictions]           # already rank-sorted
        pick = order[0]
        winner = finish[0]
        rows.append({
            "race": rno, "field": b.runners,
            "pick": pick, "pick_horse": a.predictions[0].horse,
            "pick_win_pct": a.predictions[0].win_pct,
            "winner": winner,
            "won": pick == winner,
            "pick_placed": pick in finish[:3],
            "winner_in_top3": winner in order[:3],
            "winner_rank": (order.index(winner) + 1) if winner in order else None,
            "exacta": order[:2] == finish[:2],
            "trifecta": order[:3] == finish[:3],
            "top3_all_placed": set(order[:3]) == set(finish[:3]),
            "order": order, "finish": finish,
        })
    return rows


def report(rows):
    n = len(rows)
    if not n:
        print("nothing to score")
        return
    print(f"{'R':>2}  {'field':>5}  {'pick':>4}  {'horse':<20} {'win%':>5}  "
          f"{'won':>3}  {'plc':>3}  {'winner':>6}  {'its rank':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['race']:>2}  {r['field']:>5}  {r['pick']:>4}  "
              f"{r['pick_horse'][:20]:<20} {r['pick_win_pct']:>5.1f}  "
              f"{'YES' if r['won'] else ' no':>3}  "
              f"{'YES' if r['pick_placed'] else ' no':>3}  "
              f"{r['winner']:>6}  {str(r['winner_rank']):>8}")
    print("-" * 78)
    w = sum(r["won"] for r in rows)
    p = sum(r["pick_placed"] for r in rows)
    t3 = sum(r["winner_in_top3"] for r in rows)
    ex = sum(r["exacta"] for r in rows)
    tf = sum(r["trifecta"] for r in rows)
    fields = [r["field"] for r in rows]
    rand_w = sum(1 / f for f in fields) / n
    rand_p = sum(min(3 / f, 1.0) for f in fields) / n
    rand_t3 = sum(min(3 / f, 1.0) for f in fields) / n
    print(f"races                {n}")
    print(f"top pick won         {w}/{n} = {100*w/n:.0f}%   "
          f"(random pick: {100*rand_w:.0f}%)")
    print(f"top pick placed      {p}/{n} = {100*p/n:.0f}%   "
          f"(random pick: {100*rand_p:.0f}%)")
    print(f"winner in model top3 {t3}/{n} = {100*t3/n:.0f}%   "
          f"(random: {100*rand_t3:.0f}%)")
    print(f"exacta (exact 1-2)   {ex}/{n}")
    print(f"trifecta (exact 1-3) {tf}/{n}")
    ranks = [r["winner_rank"] for r in rows if r["winner_rank"]]
    if ranks:
        print(f"mean rank of actual winner  {sum(ranks)/len(ranks):.1f} "
              f"(random would be {sum(fields)/n/2 + 0.5:.1f})")


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for path, res in RESULTS.items():
        print(f"===== {path} =====")
        report(run(path if "/" in path else "C:/Users/Admin/Downloads/" + path, res, sims))
