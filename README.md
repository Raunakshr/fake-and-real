# Fake News NLP Project

This repository contains scaffolding for a fake news detection project using NLP and deep learning techniques.

## Structure

```
 data/               # datasets (read-only)
 notebooks/          # Jupyter notebooks
 models/             # saved models and artifacts
 app/                # Django prototype (placeholder)
 docs/               # dissertation and figures
 scripts/            # training and evaluation scripts
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Django Prototype

```bash
cd app
python manage.py migrate
python manage.py runserver
```

## Notes

The project is under active development. Many components are placeholders and require further implementation.
