from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from weather_calibration.features import prepare_training_frame
from weather_calibration.modeling import save_training_outputs, train_calibrator
from weather_calibration.synthetic import write_synthetic_weather_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic weather forecast calibration demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_data = subparsers.add_parser("make-data", help="Generate a synthetic CSV dataset.")
    make_data.add_argument("--output", default="data/synthetic_weather.csv")
    make_data.add_argument("--days", type=int, default=900)
    make_data.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Train and evaluate the calibration model.")
    train.add_argument("--input", default="data/synthetic_weather.csv")
    train.add_argument("--report", default="reports/model_metrics.json")
    train.add_argument("--model-output", default=None)
    train.add_argument("--test-size", type=float, default=0.2)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-data":
        path = write_synthetic_weather_csv(args.output, days=args.days, seed=args.seed)
        print(f"Wrote synthetic dataset to {path}")
        return

    if args.command == "train":
        raw = pd.read_csv(args.input)
        frame = prepare_training_frame(raw)
        result = train_calibrator(frame, test_size=args.test_size)
        save_training_outputs(
            result,
            report_path=args.report,
            model_path=args.model_output,
        )
        print(asdict(result.report))
        return

    parser.error(f"Unknown command: {args.command}")
