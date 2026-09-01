"""Score the rating against the races whose results we know.

Three races is not evidence about the model's quality - it is a regression
harness.  It exists so that a change to the parser or the rating shows up as a
number instead of a vibe.  Run it with `python backtest.py`.
"""

from __future__ import annotations

import io
import math
import os

import rating
import rs_parser

HERE = os.path.dirname(os.path.abspath(__file__))

# fixture -> finishing order (box numbers), as reported by the user
RESULTS: dict[str, list[int]] = {
    "bulli_r6_2026-09-01.txt": [4, 1, 5],
    "qlakeside_r7_2026-09-01.txt": [4, 8, 6],
    "horsham_r8_2026-09-01.txt": [6, 1, 3, 7],
}


def load(name: str) -> rs_parser.Race:
    path = os.path.join(HERE, "fixtures", name)
    return rs_parser.parse(io.open(path, encoding="utf-8").read())


def main() -> int:
    tot_blend = tot_model = tot_market = tot_unif = 0.0
    hits = 0
    print(f"{'RACE':<26}{'PICK':<19}{'WON':<19}{'blend':>8}{'model':>8}{'market':>8}{'unif':>8}")
    for name, order in RESULTS.items():
        race = load(name)
        rated, _ = rating.rate(race)
        by_box = {x.tab: x for x in rated}
        win = by_box[order[0]]
        pick = rated[0]
        hits += pick.tab == order[0]
        n = len(rated)
        lb = -math.log(max(win.p_final, 1e-9))
        lm = -math.log(max(win.p_model, 1e-9))
        lk = -math.log(max(win.p_market or 1.0 / n, 1e-9))
        lu = -math.log(1.0 / n)
        tot_blend += lb
        tot_model += lm
        tot_market += lk
        tot_unif += lu
        tag = name.split("_")[0] + " " + name.split("_")[1]
        print(f"{tag:<26}{pick.name[:17]:<19}{win.name[:17]:<19}"
              f"{lb:>8.4f}{lm:>8.4f}{lk:>8.4f}{lu:>8.4f}")
    k = len(RESULTS)
    print(f"\n{'mean log loss':<64}{tot_blend/k:>8.4f}{tot_model/k:>8.4f}"
          f"{tot_market/k:>8.4f}{tot_unif/k:>8.4f}")
    print(f"picks: {hits} from {k}")
    edge = (tot_market - tot_blend) / k
    print(f"blend vs market: {edge:+.4f} nats/race "
          f"({'ahead' if edge > 0 else 'behind'}) - n={k}, which is far too few "
          "to mean anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
