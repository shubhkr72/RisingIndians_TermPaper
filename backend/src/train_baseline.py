from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.data_utils import load_raw_dataset, split_dataset


def train_baselines(raw_dir: str = "data/raw", out_dir: str = "models"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(raw_dir)
    train_df, _, test_df = split_dataset(df)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    x_train = tfidf.fit_transform(train_df["clean_text"])
    x_test = tfidf.transform(test_df["clean_text"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    models = {
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(),
        "logistic_regression": LogisticRegression(max_iter=400),
    }

    print("\nBaseline evaluation:")
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(f"{name:20s} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        joblib.dump(model, out_path / f"{name}.joblib")

    joblib.dump(tfidf, out_path / "tfidf_vectorizer.joblib")
    print(f"Saved baseline artifacts to {out_path.resolve()}")


if __name__ == "__main__":
    train_baselines()
