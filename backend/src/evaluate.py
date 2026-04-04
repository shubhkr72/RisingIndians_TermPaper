from __future__ import annotations

from predictor import ModelPredictor
from src.data_utils import load_raw_dataset, split_dataset


def evaluate_all():
    df = load_raw_dataset("data/raw")
    _, _, test_df = split_dataset(df)
    predictor = ModelPredictor(models_dir="models")

    print("Evaluation on test split:")
    for model_name in predictor.available_models():
        metrics = predictor.evaluate_model(test_df["clean_text"], test_df["label"], model_name)
        print(
            f"{model_name:18s} acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    evaluate_all()