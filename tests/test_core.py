from __future__ import annotations

import numpy as np
import pandas as pd

from racing_ev.features import build_feature_frame
from racing_ev.model import score_race
from racing_ev.odds import add_value_columns, devig_probabilities
from racing_ev.parser import parse_race


HORSE = r"""
# Test Park Form Guide (Race 2) | Enhanced Form
## Thursday, 03rd September 2026
16:00
(local)
## Test Handicap
Type: **HCP** Fastest Time: **1:10.00 Example**
AUD $10,000
1200m TURF GOOD
| Tab | Horse | WT | BP | Jockey | JRat | Trainer | TRat | X | Odds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [ALPHA](x) | 58.0 | 1 | [A RIDER](x) | 3.0 | [A TRAINER](x) | 3.0 | x | 2.50 |
| 2 | [BETA](x) | 57.0 | 4 | [B RIDER](x) | 2.0 | [B TRAINER](x) | 2.0 | x | 4.00 |
1
**$2.50**
[**ALPHA**](x) **4yo B Gelding (BP: 1) 58.0kg**
**Jockey**
A RIDER
**Last50**
20%-40%-50
**Trainer**
A TRAINER
**Last50**
16%-36%-50
**Filters**
**Car**
2-3-10
**Dist**
1-2-5
**Good**
2-2-8
**Facts**
**DLS**
14
##### **Days Since Last Run: 14 days (3U)**
**1 of 8**
**10d**
**80**
**OHR**
**01 Aug 2026** **TEST PARK (AUSTRALIA):** **Margin** **1L Distance** **1200m Surface** **T SOT** **G Class** **HCP API** **2.0 Race Time** **1:10.00 Jockey** **A RIDER Weight** **58 BP** **1 SP** **3.0 Trainer** **A TRAINER Track Direction** **Clockwise**
---
2
**$4.00**
[**BETA**](x) **4yo B Mare (BP: 4) 57.0kg**
**Jockey**
B RIDER
**Last50**
8%-20%-50
**Trainer**
B TRAINER
**Last50**
6%-18%-50
**Filters**
**Car**
0-2-12
**Dist**
0-1-5
**Good**
0-1-8
**Facts**
**DLS**
21
##### **Days Since Last Run: 21 days (4U)**
**5 of 8**
**14d**
**65**
**OHR**
**01 Aug 2026** **TEST PARK (AUSTRALIA):** **Margin** **5L Distance** **1200m Surface** **T SOT** **G Class** **HCP API** **2.0 Race Time** **1:10.00 Jockey** **B RIDER Weight** **57 BP** **4 SP** **6.0 Trainer** **B TRAINER Track Direction** **Clockwise**
"""


PLAIN_BROWSER = r"""
Test Park Form Guide (Race 2) | Enhanced Form
Thursday, 03rd September 2026
16:00
(local)
Test Handicap
Type: HCP Fastest Time: 1:10.00 Example
AUD $10,000
1200m TURF GOOD
1
2.50
ALPHA 4yo B Gelding (BP: 1) 58.0kg
Jockey
A RIDER
Last50
20%-40%-50
Trainer
A TRAINER
Last50
16%-36%-50
Filters
Car
2-3-10
Dist
1-2-5
Good
2-2-8
Facts
DLS
14
Days Since Last Run: 14 days (3U)
1 of 8
10d
80
OHR
01 Aug 2026 (33d ago) TEST PARK (AUSTRALIA): Margin 1L Distance 1200m Surface T SOT G Class HCP API 2.0 Race Time 1:10.00 Jockey A RIDER Weight 58 BP 1 SP $3.0 Trainer A TRAINER Track Direction Clockwise
2
4.00
BETA 4yo B Mare (BP: 4) 57.0kg
Jockey
B RIDER
Last50
8%-20%-50
Trainer
B TRAINER
Last50
6%-18%-50
Filters
Car
0-2-12
Dist
0-1-5
Good
0-1-8
Facts
DLS
21
Days Since Last Run: 21 days (4U)
5 of 8
14d
65
OHR
01 Aug 2026 (33d ago) TEST PARK (AUSTRALIA): Margin 5L Distance 1200m Surface T SOT G Class HCP API 2.0 Race Time 1:10.00 Jockey B RIDER Weight 57 BP 4 SP $6.0 Trainer B TRAINER Track Direction Clockwise
"""


def test_parser_and_probability_sum() -> None:
    card = parse_race(HORSE)
    assert card.discipline == "thoroughbred"
    assert card.race["field_size"] == 2
    assert len(card.histories) == 2
    features = build_feature_frame(card)
    result = score_race(features, card.discipline)
    assert np.isclose(result.table["win_probability"].sum(), 1.0)
    assert result.table.iloc[0]["runner"] == "ALPHA"


def test_plain_browser_clipboard_fallback() -> None:
    card = parse_race(PLAIN_BROWSER)
    assert card.race["field_size"] == 2
    assert [r["runner"] for r in card.runners] == ["ALPHA", "BETA"]
    assert card.runners[0]["market_odds"] == 2.50
    assert card.runners[1]["market_odds"] == 4.00
    assert len(card.histories) == 2
    assert card.race["parser_mode"] == "browser-clipboard-fallback"


def test_devig_and_ev_formula() -> None:
    odds = np.array([2.0, 3.0, 5.0])
    p, overround = devig_probabilities(odds, "proportional")
    assert np.isclose(p.sum(), 1.0)
    assert overround > 0
    frame = pd.DataFrame({
        "runner": ["A", "B", "C"],
        "market_odds": odds,
        "win_probability": [0.55, 0.30, 0.15],
        "data_quality": [1.0, 1.0, 1.0],
    })
    out, _ = add_value_columns(frame, method="proportional")
    assert np.isclose(out.loc[0, "ev_per_unit"], 0.10)


def test_training_rows_must_not_mix_disciplines() -> None:
    from racing_ev.training import train_models

    frame = pd.DataFrame({
        "race_id": [f"r{i}" for i in range(80) for _ in range(2)],
        "race_date": [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(80) for _ in range(2)],
        "discipline": ["thoroughbred", "greyhound"] * 80,
        "won": [1, 0] * 80,
        **{c: [0.0] * 160 for c in [
            "recent_finish_score", "recent_win_signal", "recent_place_signal",
            "margin_signal", "speed_signal", "rating_signal", "suitability_signal",
            "connection_signal", "fitness_signal", "setup_signal",
        ]},
    })
    try:
        train_models(frame)
    except ValueError as exc:
        assert "one discipline at a time" in str(exc).lower()
    else:
        raise AssertionError("Mixed disciplines should be rejected")
