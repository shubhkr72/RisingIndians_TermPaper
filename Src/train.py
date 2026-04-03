# src/train.py

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classical ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Deep Learning
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text


# =========================
# CONFIG
# =========================
MAX_LEN = 300
VOCAB_SIZE = 20000
EPOCHS = 5
BATCH_SIZE = 32


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")

fake = pd.read_csv("Data/Raw/Fake.csv")
real = pd.read_csv("Data/Raw/True.csv")

fake['label'] = 1
real['label'] = 0

df = pd.concat([fake, real])
df = df[['text', 'label']]
df = df.sample(frac=1).reset_index(drop=True)

print("Dataset size:", len(df))


# =========================
# PREPROCESSING
# =========================
print("Cleaning text...")
df['text'] = df['text'].apply(clean_text)


# =========================
# TRAIN TEST SPLIT
# =========================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)


# =====================================================
# 🔹 BASELINE MODELS (TF-IDF)
# =====================================================
print("\nTraining Baseline Models...")

tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# 1. Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

# 2. SVM
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

# 3. Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)

print("\nBaseline Results:")
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


# =====================================================
# 🔹 DEEP LEARNING MODEL (Bi-LSTM + CNN)
# =====================================================
print("\nTraining Deep Learning Model...")

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)


# =========================
# MODEL (Bi-LSTM + CNN)
# =========================
inp = Input(shape=(MAX_LEN,))

emb = layers.Embedding(VOCAB_SIZE, 128)(inp)

# Bi-LSTM branch
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(emb)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dropout(0.3)(x)

# CNN branch
c = layers.Conv1D(128, 3, activation='relu')(emb)
c = layers.GlobalMaxPooling1D()(c)

# Fusion
m = layers.Concatenate()([x, c])
m = layers.Dense(64, activation='relu')(m)
m = layers.Dropout(0.3)(m)

out = layers.Dense(1, activation='sigmoid')(m)

model = Model(inp, out)


# =========================
# COMPILE
# =========================
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# =========================
# TRAIN
# =========================
history = model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)


# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_test_pad, y_test)

print("\nDeep Learning Accuracy:", acc)


# =========================
# SAVE EVERYTHING
# =========================
os.makedirs("models", exist_ok=True)

model.save("models/model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nAll models saved successfully!")