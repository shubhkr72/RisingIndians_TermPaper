# predictor.py

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text


# =========================
# CONFIG
# =========================
MAX_LEN = 300


# =========================
# LOAD MODEL + TOKENIZER
# =========================
print("Loading model...")

model = load_model("models/model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# =========================
# PREDICT FUNCTION
# =========================
def predict(text: str):
    # preprocess
    text = clean_text(text)

    # tokenize
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    # predict
    prob = model.predict(pad)[0][0]

    fake_prob = float(prob)
    real_prob = float(1 - prob)

    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = float(prob if prob > 0.5 else (1 - prob))

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4)
    }