# Fake News Detection (Term Paper Implementation)

This project implements the term paper pipeline for fake news detection using:

- Baselines: Naive Bayes, SVM, Logistic Regression (TF-IDF features)
- Deep model: BiLSTM + CNN
- API: Flask REST endpoint (`POST /api/v1/predict`)

## Dataset

Put these files in `data/raw/`:

- `Fake.csv`
- `True.csv`

Both are commonly available in the ISOT/Kaggle fake news datasets.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train All Models

```bash
python -m src.train
```

This runs:

1. `src/train_baseline.py`
2. `src/train_dl.py`

## Evaluate

```bash
python -m src.evaluate
```

## Run Flask API

```bash
python app.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Prediction:

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Government confirms major new policy update.\",\"title\":\"Breaking\",\"model\":\"hybrid\"}"
```

`model` can be one of:

- `naive_bayes`
- `svm`
- `logistic_regression`
- `bilstm_cnn`
- `hybrid` (average probability from available models)

![alt text](image.png)
