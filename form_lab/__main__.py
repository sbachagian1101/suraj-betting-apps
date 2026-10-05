from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import inventory
from .schema import COUNTRIES, DISCIPLINES, audit, read_jsonl
from .storage import append
from .training import Bundle, train


def main():
    parser = argparse.ArgumentParser(description="Form-only research engine; isolated from the existing predictor")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inventory", help="Read-only inventory of an existing app directory")
    p.add_argument("root")
    p = sub.add_parser("audit", help="Validate canonical full-field JSONL")
    p.add_argument("data")
    p = sub.add_parser("ingest", help="Append canonical snapshots to the separate ledger")
    p.add_argument("data")
    p.add_argument("--ledger", required=True)
    p = sub.add_parser("train")
    p.add_argument("data")
    p.add_argument("--output", required=True)
    p.add_argument("--country", choices=sorted(COUNTRIES), required=True)
    p.add_argument("--discipline", choices=sorted(DISCIPLINES), required=True)
    for name in ("blend-start", "calibration-start", "test-start"):
        p.add_argument("--" + name, required=True, help="Timezone-aware ISO timestamp, preferably UTC midnight")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--min-races", type=int, default=20, help="Engineering floor per block, not evidence of statistical adequacy")
    p = sub.add_parser("predict")
    p.add_argument("cards")
    p.add_argument("--model", required=True)
    p.add_argument("--history", required=True, help="Canonical past full-field archive used for connection and horse state")
    p.add_argument("--ledger", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.root)
    elif args.command == "audit":
        result = audit(read_jsonl(args.data))
    elif args.command == "ingest":
        records = read_jsonl(args.data)
        ids = [append(args.ledger, "snapshot", r) for r in records]
        result = {"records": len(ids), "ids": ids}
    elif args.command == "train":
        manifest, report = train(read_jsonl(args.data), args.output, country=args.country,
                                 discipline=args.discipline, blend_start=args.blend_start,
                                 calibration_start=args.calibration_start, test_start=args.test_start,
                                 iterations=args.iterations, min_races=args.min_races)
        result = {"model_version": manifest["model_version"], "output": args.output,
                  "test_metrics": report["test_metrics"]}
    else:
        if Path(args.output).exists():
            raise ValueError("Prediction output already exists; choose a new filename")
        model = Bundle(args.model)
        history = read_jsonl(args.history) if args.history else []
        predictions = [model.predict(r, history) for r in read_jsonl(args.cards)]
        for prediction in predictions:
            append(args.ledger, "prediction", prediction)
        Path(args.output).write_text("\n".join(json.dumps(p, allow_nan=False) for p in predictions) + "\n")
        result = {"predictions": len(predictions), "output": args.output}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
