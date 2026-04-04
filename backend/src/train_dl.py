from __future__ import annotations

import pickle
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.data_utils import load_raw_dataset, split_dataset
from src.models.bilstm import build_bilstm_cnn

MAX_LEN = 300
VOCAB_SIZE = 30000


def train_deep_model(raw_dir: str = "data/raw", out_dir: str = "models", epochs: int = 4):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(raw_dir)
    train_df, val_df, test_df = split_dataset(df)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df["clean_text"]), maxlen=MAX_LEN
    )
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_df["clean_text"]), maxlen=MAX_LEN)
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(test_df["clean_text"]), maxlen=MAX_LEN
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    model = build_bilstm_cnn(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    model.compile(
        optimizer=Adam(learning_rate=2e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=[stop],
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"BiLSTM+CNN test accuracy: {acc:.4f}, loss: {loss:.4f}")

    keras_path = out_path / "bilstm_cnn.keras"
    model.save(keras_path)
    with open(out_path / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved deep model artifacts to {out_path.resolve()}")


if __name__ == "__main__":
    train_deep_model()
