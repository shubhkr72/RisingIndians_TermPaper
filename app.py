# src/evaluate.py

import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import clean_text


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
# LOAD DATA
# =========================
print("Loading dataset...")

fake = pd.read_csv("data/raw/Fake.csv")
real = pd.read_csv("data/raw/True.csv")

fake['label'] = 1
real['label'] = 0

df = pd.concat([fake, real])
df = df[['text', 'label']]

# optional: use only test subset
df = df.sample(frac=0.2, random_state=42)


# =========================
# PREPROCESS
# =========================
print("Cleaning text...")

df['text'] = df['text'].apply(clean_text)


# =========================
# TOKENIZE
# =========================
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN)

y_true = df['label'].values


# =========================
# PREDICT
# =========================
print("Predicting...")

y_prob = model.predict(X)
y_pred = (y_prob > 0.5).astype(int)


# =========================
# METRICS
# =========================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n===== Evaluation Results =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))