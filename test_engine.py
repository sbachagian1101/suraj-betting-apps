from pathlib import Path
import unittest

from engine import analyse_race_text, parse_last_start, parse_race_text


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = Path(__file__).with_name("sample_race.txt").read_text(encoding="utf-8")

    def test_sample_parses(self):
        race, runners, warnings, summary = parse_race_text(self.sample)
        self.assertEqual(race.race_number, 1)
        self.assertEqual(race.distance_m, 1650)
        self.assertEqual(race.surface, "Turf")
        self.assertEqual(race.going, "Soft")
        self.assertEqual(race.field_size, 12)
        self.assertEqual(runners[0].horse, "EMERGING STAR")
        self.assertEqual(runners[-1].horse, "PERFECTO MOMENTS")
        self.assertIn("Parsed 12 runners", summary)


    def test_ordinal_last_start_is_parsed(self):
        last = parse_last_start(
            "1st-0.8L-LATE7-1900m-4UHCP CL4g--$10--57kg"
        )
        self.assertEqual(last.finish, 1)
        self.assertAlmostEqual(last.margin_l, 0.8)
        self.assertEqual(last.venue_code, "LATE7")
        self.assertEqual(last.distance_m, 1900)
        self.assertEqual(last.class_number, 4)
        self.assertEqual(last.odds, 10.0)
        self.assertEqual(last.weight_kg, 57.0)

    def test_ordinal_win_contributes_to_prediction(self):
        headers = "\t".join([
            "Tab", "Horse", "Form L5", "BP", "12m%", "Car%", "Dist%",
            "DLR", "Crs%", "JRat", "TRat", "PM 12m", "GD%", "Turf%",
            "AW%", "SH%", "PM Car", "LS Det", "CD%",
        ])
        row = "\t".join([
            "2", "MONTE LINAS", "1x041", "6", "15-31-13", "13-29-31",
            "7-21-14", "42", "100-100-1", "2.2", "3.0", "$1,780",
            "18-36-11", "16-32-19", "8-25-12", "0-14-7", "$1,508",
            "1st-0.8L-LATE7-1900m-4UHCP CL4g--$10--57kg", "0-0-0",
        ])
        text = "\n".join([
            "2\tPRIX LOUIS DU CHEYRON [26,987]",
            "\tType : 4U HCP CL4",
            "16:55\t1600m, TURF S",
            "",
            headers,
            row,
        ])
        race, runners, _, _ = parse_race_text(text)
        self.assertEqual(race.class_number, 4)
        self.assertEqual(runners[0].last_start.finish, 1)

        analysis = analyse_race_text(text, simulations=1000)
        factors = {d.key: d for d in analysis.predictions[0].factor_details}
        self.assertTrue(factors["last_start"].available)
        self.assertGreater(factors["last_start"].score, 85.0)
        self.assertIn("Finished 1", factors["last_start"].note)
        self.assertTrue(factors["class_move"].available)
        self.assertIn("Same CL4", factors["class_move"].note)

    def test_prediction_probabilities(self):
        analysis = analyse_race_text(self.sample, simulations=3000)
        self.assertEqual(len(analysis.predictions), 12)
        self.assertAlmostEqual(sum(p.win_pct for p in analysis.predictions), 100.0, delta=0.2)
        self.assertTrue(all(0 <= p.top3_pct <= 100 for p in analysis.predictions))
        self.assertEqual(analysis.predictions[0].rank, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
