# Fake News NLP Project

This repository contains an end-to-end fake news detection project using NLP
and deep learning techniques.

## Structure

```
 data/               # datasets (read-only)
 notebooks/          # Jupyter notebooks
 models/             # saved models and artifacts
 app/                # Django prototype
 docs/               # dissertation and figures
 scripts/            # training and evaluation scripts
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Classical model training and evaluation:

```bash
python scripts/train.py --model classical
python scripts/evaluate.py
```

To train the example deep model instead, pass ``--model deep`` to
``train.py``.

## Django Prototype

After training and exporting the pipeline, run the web demo:

```bash
cd app
python manage.py migrate
python manage.py runserver
```

The root page provides a simple form, while ``/api/predict`` accepts JSON
payloads of the form ``{"text": "..."}``.
