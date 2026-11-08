
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE / "data/raw/cicids2017/MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

MODEL_PATH = BASE / "models/cicids2017_expanded_model.pkl"

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

SAMPLE_SIZE = 2000


def main():
    print("Loading expanded model...")
    model = joblib.load(MODEL_PATH)

    print("Loading PortScan dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    print("\nAvailable labels:")
    print(df["Label"].value_counts())

    benign = df[df["Label"] == "BENIGN"]
    attacks = df[df["Label"] == "PortScan"]

    if benign.empty or attacks.empty:
        raise ValueError("BENIGN or PortScan records were not found.")

    test_df = pd.concat([
        benign.sample(
            n=min(SAMPLE_SIZE, len(benign)),
            random_state=42,
        ),
        attacks.sample(
            n=min(SAMPLE_SIZE, len(attacks)),
            random_state=42,
        ),
    ], ignore_index=True)

    X = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    y_true = (
        test_df["Label"] != "BENIGN"
    ).astype(int).to_numpy()

    predictions = model.predict(X)

    print(f"\nRecords evaluated: {len(test_df)}")
    print(
        f"Accuracy: {accuracy_score(y_true, predictions) * 100:.2f}%"
    )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    print("\nDetection rate by category:")
    for label in ["BENIGN", "PortScan"]:
        mask = test_df["Label"].to_numpy() == label
        rate = (predictions[mask] == (label == "PortScan")).mean()
        print(f"{label}: {rate * 100:.2f}%")

    print(
        "\nNote: PortScan is not DoS/DDoS. This is an external "
        "attack-category test, not a live-traffic assessment."
    )


if __name__ == "__main__":
    main()
