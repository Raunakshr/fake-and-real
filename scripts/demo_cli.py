"""Command line interface for quick predictions.

This script loads the exported ``pipeline.joblib`` and performs a single
prediction for the supplied text snippet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pipeline.joblib"
LABELS = {0: "fake", 1: "real"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake news detector CLI")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba([args.text])[0]
    label_idx = int(proba.argmax())
    label = LABELS.get(label_idx, str(label_idx))
    print(f"Input: {args.text}\nPrediction: {label} (p={proba[label_idx]:.3f})")


if __name__ == "__main__":
    main()
