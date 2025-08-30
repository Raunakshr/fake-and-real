"""Evaluation script for fake news detection models.


This utility loads a saved pipeline and reports standard classification
metrics on the held‑out test set derived from the repository datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_dataset() -> pd.DataFrame:
    fake = pd.read_csv(DATA_DIR / "Fake.csv.zip")
    fake["label"] = "fake"
    true = pd.read_csv(DATA_DIR / "True.csv.zip")
    true["label"] = "real"
    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df[["text", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved models")
    parser.add_argument(
        "--model-path", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "pipeline.joblib")
    )
    args = parser.parse_args()

    df = load_dataset()
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = joblib.load(args.model_path)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


if __name__ == "__main__":

TODO: Implement evaluation routine.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models")
    parser.add_argument('--model-path', type=str, default='../models/classical_best.pkl')
    args = parser.parse_args()
    # TODO: Add evaluation logic
    print(f"Evaluating model at {args.model_path} - TODO")


if __name__ == '__main__':

    main()
