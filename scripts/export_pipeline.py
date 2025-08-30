"""Utility to export the trained pipeline for inference.

This script loads the individual artefacts produced during training and
combines them into a single ``pipeline.joblib`` file that the inference code
can consume.  If the pipeline already exists it will be overwritten.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    clf = joblib.load(MODEL_DIR / "classical_best.pkl")
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    print(f"Pipeline exported to {MODEL_DIR / 'pipeline.joblib'}")


if __name__ == "__main__":

TODO: Implement export logic.
"""


def main():
    # TODO: Load models and save combined pipeline
    print("Exporting pipeline - TODO")


if __name__ == '__main__':

    main()
