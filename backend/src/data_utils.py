from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text, combine_title_text


def load_raw_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    raw_path = Path(raw_dir)
    fake_path = raw_path / "Fake.csv"
    real_path = raw_path / "True.csv"

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 1
    real["label"] = 0

    df = pd.concat([fake, real], ignore_index=True)
    if "title" not in df.columns:
        df["title"] = ""
    if "text" not in df.columns:
        raise ValueError("Input data must contain a 'text' column.")

    df["input_text"] = df.apply(
        lambda row: combine_title_text(row.get("title", ""), row.get("text", "")),
        axis=1,
    )
    df["clean_text"] = df["input_text"].apply(clean_text)
    df = df[["input_text", "clean_text", "label"]].dropna()
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )
    return train_df, val_df, test_df
