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

## Documentation

### Prerequisites

Building the dissertation PDF requires a LaTeX distribution such as [TeX Live](https://www.tug.org/texlive/).

### Building the PDF

```bash
cd docs
pdflatex dissertation.tex
bibtex dissertation
pdflatex dissertation.tex
pdflatex dissertation.tex
```

To automate the build, run:

```bash
cd docs
make
```
