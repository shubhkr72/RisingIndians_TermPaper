from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.preprocess import clean_text

_HYBRID_CANDIDATES = (
    "naive_bayes",
    "svm",
    "logistic_regression",
    "bilstm_cnn",
)

MAX_LEN = 300


class ModelPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.tfidf = None
        self.tokenizer = None
        self.baselines: Dict[str, object] = {}
        self.bilstm_cnn = None
        self.pad_sequences = None
        self._load_artifacts()

    def _load_artifacts(self):
        tfidf_path = self.models_dir / "tfidf_vectorizer.joblib"
        if tfidf_path.exists():
            self.tfidf = joblib.load(tfidf_path)
            for name in ["naive_bayes", "svm", "logistic_regression"]:
                model_path = self.models_dir / f"{name}.joblib"
                if model_path.exists():
                    self.baselines[name] = joblib.load(model_path)

        tokenizer_path = self.models_dir / "tokenizer.pkl"
        bilstm_path = self.models_dir / "bilstm_cnn.keras"
        if not bilstm_path.exists():
            bilstm_path = self.models_dir / "bilstm_cnn.h5"
        if tokenizer_path.exists() and bilstm_path.exists():
            try:
                from tensorflow.keras.models import load_model
                from tensorflow.keras.preprocessing.sequence import pad_sequences

                with open(tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
                self.bilstm_cnn = load_model(bilstm_path, compile=False)
                self.pad_sequences = pad_sequences
            except Exception:
                self.bilstm_cnn = None
                self.tokenizer = None
                self.pad_sequences = None

    def available_models(self) -> List[str]:
        names = list(self.baselines.keys())
        if self.bilstm_cnn is not None:
            names.append("bilstm_cnn")
        if names:
            names.append("hybrid")
        return names

    @staticmethod
    def _format_output(fake_probability: float):
        fake_probability = float(np.clip(fake_probability, 0.0, 1.0))
        real_probability = 1.0 - fake_probability
        label = "FAKE" if fake_probability >= 0.5 else "REAL"
        confidence = fake_probability if label == "FAKE" else real_probability
        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "real_probability": round(float(real_probability), 4),
            "fake_probability": round(float(fake_probability), 4),
        }

    def _predict_proba_single(self, clean_input_text: str, model_name: str) -> float:
        if model_name in self.baselines:
            if self.tfidf is None:
                raise ValueError("TF-IDF vectorizer not found.")
            x = self.tfidf.transform([clean_input_text])
            model = self.baselines[model_name]
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(x)[0][1])
            pred = int(model.predict(x)[0])
            return 0.99 if pred == 1 else 0.01

        if model_name == "bilstm_cnn":
            if self.bilstm_cnn is None or self.tokenizer is None:
                raise ValueError("BiLSTM+CNN model artifacts not found.")
            seq = self.tokenizer.texts_to_sequences([clean_input_text])
            x = self.pad_sequences(seq, maxlen=MAX_LEN)
            return float(self.bilstm_cnn.predict(x, verbose=0)[0][0])

        if model_name == "hybrid":
            candidates = []
            for candidate in _HYBRID_CANDIDATES:
                try:
                    candidates.append(self._predict_proba_single(clean_input_text, candidate))
                except Exception:
                    continue
            if not candidates:
                raise ValueError("No trained model is available for hybrid prediction.")
            return float(np.mean(candidates))

        raise ValueError(f"Unsupported model: {model_name}")

    def predict(self, text: str, title: str = "", model_name: str = "hybrid"):
        combined = f"{title} {text}".strip()
        cleaned = clean_text(combined)
        fake_probability = self._predict_proba_single(cleaned, model_name)
        output = self._format_output(fake_probability)
        output["model"] = model_name
        return output

    def evaluate_model(self, texts, labels, model_name: str):
        probs = [self._predict_proba_single(text, model_name) for text in texts]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }


_PREDICTOR = ModelPredictor("models")


def predict(text: str, title: str = "", model_name: str = "hybrid"):
    return _PREDICTOR.predict(text=text, title=title, model_name=model_name)
