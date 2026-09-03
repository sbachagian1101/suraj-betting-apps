"""CLI: train a Racing EV Lab model from labelled feature rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from racing_ev.training import train_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    result = train_models(frame)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.artifact, args.output_model)
    for key, value in result.metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
