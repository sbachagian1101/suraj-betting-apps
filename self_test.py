from pathlib import Path
import json

from bet365_parser import Bet365TextParser
from bet365_model import predict_race
from storage import load_model_state
from results_parser import parse_multi_meeting_results, validate_result_for_race

BASE = Path(__file__).resolve().parent
parsed = Bet365TextParser().parse_file(BASE / "samples" / "Bet365_predictions_sample.txt")
assert parsed["race_count"] == 39, parsed["race_count"]
assert len(parsed["meetings"]) == 5
results = parse_multi_meeting_results((BASE / "seed_results.txt").read_text(encoding="utf-8"), [m["meeting"] for m in parsed["meetings"]])
assert sum(len(v) for v in results.values()) == 35
state = load_model_state(BASE / "data")
for meeting in parsed["meetings"]:
    for race in meeting["races"]:
        prediction = predict_race(race, state, simulations=500)
        assert len(prediction["rows"]) == race["active_field_size"]
        assert abs(sum(row["win_probability"] for row in prediction["rows"]) - 1.0) < 1e-10
        result = results.get(meeting["meeting"], {}).get(race["race_no"])
        if result:
            ok, message = validate_result_for_race(race, result)
            assert ok, message
print(json.dumps({
    "meetings": len(parsed["meetings"]),
    "races": parsed["race_count"],
    "labelled_races": sum(len(v) for v in results.values()),
    "pending_port_hedland_races": 4,
    "training_races": state.get("training_races"),
    "training_pairs": state.get("training_pairs"),
    "learned_influence": state.get("learned_influence"),
    "status": "OK",
}, indent=2))
