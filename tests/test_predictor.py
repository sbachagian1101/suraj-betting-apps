from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from bet365_parser import Bet365TextParser
from bet365_model import default_model_state, predict_race, train_model, training_record_from_race
from results_parser import parse_multi_meeting_results, validate_result_for_race
from storage import load_model_state, save_training_system


class Bet365PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_text = (BASE / "samples" / "Bet365_predictions_sample.txt").read_text(encoding="utf-8")
        cls.parsed = Bet365TextParser().parse_text(cls.source_text, source_name="sample.txt")
        cls.meetings = {m["meeting"]: m for m in cls.parsed["meetings"]}
        cls.results = parse_multi_meeting_results((BASE / "seed_results.txt").read_text(encoding="utf-8"), list(cls.meetings))

    def test_multi_meeting_parse(self) -> None:
        self.assertEqual(self.parsed["race_count"], 39)
        self.assertEqual({name: len(m["races"]) for name, m in self.meetings.items()}, {
            "SANDOWN-LAKESIDE": 8, "Sunshine Coast": 8, "Wyong": 8, "Forbes": 8, "Port Hedland": 7,
        })
        declared = sum(r["declared_field_size"] for m in self.meetings.values() for r in m["races"])
        active = sum(r["active_field_size"] for m in self.meetings.values() for r in m["races"])
        self.assertEqual(declared, 485)
        self.assertEqual(active, 476)

    def test_source_and_runner_parsing(self) -> None:
        race = self.meetings["SANDOWN-LAKESIDE"]["races"][0]
        self.assertEqual(race["distance_m"], 3400)
        self.assertEqual(race["discipline"], "Hurdle")
        self.assertIn("Golden Crusader", race["suggested_play"])
        split = race["runners"][0]
        self.assertEqual(split["horse"], "SPLIT")
        self.assertEqual(split["barrier"], 5)
        self.assertEqual(split["jockey"], "WILL GORDON")
        self.assertEqual(split["historical_runs"][0]["finish"], 3)
        self.assertEqual(split["historical_runs"][0]["field_size"], 8)

    def test_scratchings(self) -> None:
        scratched = [runner for m in self.meetings.values() for race in m["races"] for runner in race["runners"] if runner["status"] == "SCRATCHED"]
        self.assertEqual(len(scratched), 9)

    def test_result_mapping_and_pending(self) -> None:
        self.assertEqual(sum(len(rows) for rows in self.results.values()), 35)
        self.assertEqual(len(self.results["Port Hedland"]), 3)
        for name, rows in self.results.items():
            for race_no, result in rows.items():
                race = self.meetings[name]["races"][race_no - 1]
                ok, message = validate_result_for_race(race, result)
                self.assertTrue(ok, f"{name} R{race_no}: {message}")

    def test_probability_distributions(self) -> None:
        state = load_model_state(BASE / "data")
        for meeting in self.meetings.values():
            for race in meeting["races"]:
                prediction = predict_race(race, state, simulations=500)
                self.assertEqual(len(prediction["rows"]), race["active_field_size"])
                self.assertAlmostEqual(sum(row["win_probability"] for row in prediction["rows"]), 1.0, places=10)
                self.assertEqual(sorted(prediction["order"]), sorted(r["number"] for r in race["runners"] if r["status"] == "ACTIVE"))

    def test_odds_mutation_invariance(self) -> None:
        altered = re.sub(r"\$\d+(?:\.\d+)?", "$9999.99", self.source_text)
        base = Bet365TextParser().parse_text(self.source_text)
        changed = Bet365TextParser().parse_text(altered)
        state = default_model_state()
        for base_meeting, changed_meeting in zip(base["meetings"], changed["meetings"]):
            for base_race, changed_race in zip(base_meeting["races"], changed_meeting["races"]):
                p1 = predict_race(base_race, state, simulations=400)
                p2 = predict_race(changed_race, state, simulations=400)
                self.assertEqual(p1["order"], p2["order"])
                for r1, r2 in zip(p1["rows"], p2["rows"]):
                    self.assertAlmostEqual(r1["win_probability"], r2["win_probability"], places=12)

    def test_training_and_json_persistence(self) -> None:
        records = []
        meeting = self.meetings["SANDOWN-LAKESIDE"]
        for race in meeting["races"][:3]:
            records.append(training_record_from_race(race, self.results["SANDOWN-LAKESIDE"][race["race_no"]]))
        state = train_model(records, epochs=30)
        self.assertEqual(state["training_races"], 3)
        self.assertGreater(state["training_pairs"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            save_training_system(data, state, records, [{"event": "test"}])
            loaded = load_model_state(data)
            self.assertEqual(loaded["training_races"], 3)
            self.assertTrue((data / "training_store.json").exists())

    def test_single_shared_input_widget(self) -> None:
        """One shared race-card box, not one per meeting.

        The original asserted `self.input_text = tk.Text(` appeared exactly
        once, which was the Tkinter way of saying it. This build is Streamlit,
        so the same requirement is checked against the widget that replaced it:
        exactly one race-card text area, keyed `b3_raw`, with the results box
        the only other one.
        """
        source = (BASE / "app.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('key="b3_raw"'), 1)
        self.assertEqual(source.count("st.text_area("), 2)
        self.assertIn('key="b3_res"', source)
        self.assertIn("Results & training", source)
        self.assertIn("model_state", source)


if __name__ == "__main__":
    unittest.main()
