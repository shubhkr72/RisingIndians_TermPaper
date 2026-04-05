from src.train_baseline import train_baselines
from src.train_dl import train_deep_model


def train_all():
    print("Step 1/2: training baseline models (NB, SVM, Logistic Regression)")
    train_baselines()

    print("\nStep 2/2: training BiLSTM + CNN deep model")
    train_deep_model()

    print("\nAll requested models were trained and saved.")


if __name__ == "__main__":
    train_all()
