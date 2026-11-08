
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
    / "Wednesday-workingHours.pcap_ISCX.csv"
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

DOS_LABELS = [
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
]


def main():
    print("Loading expanded model...")
    model = joblib.load(MODEL_PATH)

    print("Loading Wednesday dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    print("\nAvailable labels:")
    print(df["Label"].value_counts())

    # Select benign traffic and the four DoS categories.
    selected = df[
        df["Label"].isin(["BENIGN"] + DOS_LABELS)
    ].copy()

    # Sample up to 5,000 records from each category.
    parts = []

    for label in ["BENIGN"] + DOS_LABELS:
        group = selected[selected["Label"] == label]

        if not group.empty:
            parts.append(
                group.sample(
                    n=min(5000, len(group)),
                    random_state=42,
                )
            )

    test_df = pd.concat(parts, ignore_index=True)

    X = test_df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    # Binary evaluation: BENIGN = 0, any DoS category = 1.
    y_true = (
        test_df["Label"] != "BENIGN"
    ).astype(int).to_numpy()

    print("\nEvaluation label distribution:")
    print(test_df["Label"].value_counts())
    print(f"\nTotal records: {len(test_df)}")

    print("\nRunning predictions...")
    predictions = model.predict(X)

    print(
        f"\nAccuracy: {accuracy_score(y_true, predictions) * 100:.2f}%"
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

    print("\nDetection rate by actual category:")

    for label in DOS_LABELS:
        group = test_df[test_df["Label"] == label]

        if group.empty:
            continue

        features = group[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        features = features.replace(
            [np.inf, -np.inf], np.nan
        )

        predicted = model.predict(features)
        rate = (predicted == 1).mean() * 100

        print(f"{label}: {rate:.2f}%")

    print(
        "\nNote: This evaluates labeled CIC-IDS2017 traffic, "
        "not live website traffic."
    )


if __name__ == "__main__":
    main()
