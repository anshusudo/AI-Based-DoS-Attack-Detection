
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data/raw/cicids2017/MachineLearningCVE"
    / "Monday-WorkingHours.pcap_ISCX.csv"
)

MODEL_PATH = (
    BASE_DIR / "models/cicids2017_expanded_model.pkl"
)

FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
]


def main():
    print("Loading expanded model...")
    model = joblib.load(MODEL_PATH)

    print("Loading Monday traffic dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    print("\nOriginal label distribution:")
    print(df["Label"].value_counts())

    # Monday's capture is used here as a benign-only test.
    df = df[df["Label"] == "BENIGN"].copy()

    if df.empty:
        raise ValueError("No BENIGN records found.")

    # Use a reproducible sample to keep evaluation manageable.
    test_df = df.sample(
        n=min(10000, len(df)),
        random_state=42,
    )

    X = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"\nBenign records evaluated: {len(X)}")
    print("Running predictions...")

    predictions = model.predict(X)

    false_positives = int((predictions == 1).sum())
    total = len(predictions)
    false_positive_rate = false_positives / total * 100

    print("\nEvaluation results")
    print("------------------")
    print(f"Total benign records: {total}")
    print(f"Predicted BENIGN: {int((predictions == 0).sum())}")
    print(f"Predicted ATTACK: {false_positives}")
    print(f"False-positive rate: {false_positive_rate:.2f}%")

    print(
        "\nNote: This measures false positives on a sampled "
        "benign-only capture. It does not measure attack "
        "detection or prove live-network performance."
    )


if __name__ == "__main__":
    main()
