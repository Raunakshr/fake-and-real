"""Training script for fake news detection.


This module trains either a classical machine learning model or a simple
deep-learning model and persists the resulting artefacts into the ``models``
directory.  The classical track uses a TF–IDF vectoriser with logistic
regression, while the deep track trains a small LSTM network.  Random seeds
are fixed for reproducibility.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Deep learning imports are optional and imported lazily inside the function

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset() -> pd.DataFrame:
    """Load fake and real news datasets from the data directory."""

    fake = pd.read_csv(DATA_DIR / "Fake.csv.zip")
    fake["label"] = "fake"
    true = pd.read_csv(DATA_DIR / "True.csv.zip")
    true["label"] = "real"
    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df[["text", "label"]]


def train_classical(df: pd.DataFrame) -> None:
    """Train a TF–IDF + Logistic Regression pipeline."""

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipeline: Pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipeline.fit(X_train, y_train_enc)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(clf, MODEL_DIR / "classical_best.pkl")
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")

    # quick evaluation to console
    score = pipeline.score(X_test, label_encoder.transform(y_test))
    print(f"Validation accuracy: {score:.3f}")


def train_deep(df: pd.DataFrame) -> None:
    """Train a minimal LSTM network using Keras."""

    from tensorflow import keras
    from tensorflow.keras import layers

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=30000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    max_len = 200
    X_train_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=max_len
    )
    X_test_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=max_len
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    model = keras.Sequential(
        [
            layers.Embedding(input_dim=30000, output_dim=128, input_length=max_len),
            layers.LSTM(64),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_seq, y_train_enc, epochs=1, batch_size=64, validation_split=0.1)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "deep_best.h5")
    with open(MODEL_DIR / "tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

    loss, acc = model.evaluate(X_test_seq, y_test_enc, verbose=0)
    print(f"Validation accuracy: {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models for fake news detection")
    parser.add_argument("--model", choices=["classical", "deep"], default="classical")
    args = parser.parse_args()

    set_seeds()
    data = load_dataset()

    if args.model == "classical":
        train_classical(data)
    else:
        train_deep(data)


if __name__ == "__main__":

TODO: Implement full training pipeline.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train models for fake news detection")
    parser.add_argument('--model', choices=['classical', 'deep'], default='classical')
    args = parser.parse_args()
    # TODO: Add training logic
    print(f"Training {args.model} model - TODO")


if __name__ == '__main__':

    main()
