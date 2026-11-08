
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data/raw/cicids2017/MachineLearningCVE"
    / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models/cicids2017_expanded_model.pkl"
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

SAMPLE_PER_LABEL = 2000


def main():
    print("Loading expanded model...")
    model = joblib.load(MODEL_PATH)

    print("Loading Thursday Web Attacks dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    missing = [
        c for c in FEATURES + ["Label"]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\nOriginal labels:")
    print(df["Label"].value_counts())

    # Normalize labels to handle unusual characters in CSV labels.
    labels = df["Label"].str.lower()

    benign = df[labels == "benign"].copy()
    web_attacks = df[
        labels.str.startswith("web attack")
    ].copy()

    if benign.empty or web_attacks.empty:
        raise ValueError(
            "Could not find both BENIGN and Web Attack records."
        )

    # Keep all available web attack records, and sample benign records.
    benign_sample = benign.sample(
        n=min(SAMPLE_PER_LABEL, len(benign)),
        random_state=42,
    )

    test_df = pd.concat(
        [benign_sample, web_attacks],
        ignore_index=True,
    )

    X = test_df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    y_true = (
        test_df["Label"].str.lower() != "benign"
    ).astype(int).to_numpy()

    print(f"\nRecords evaluated: {len(test_df):,}")
    print("\nEvaluation categories:")
    print(test_df["Label"].value_counts())

    print("\nRunning predictions...")
    y_pred = model.predict(X)

    print(
        "\nExternal accuracy: "
        f"{accuracy_score(y_true, y_pred) * 100:.2f}%"
    )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "WEB ATTACK"],
        zero_division=0,
    ))

    print("\nDetection rate by category:")

    for label, group in test_df.groupby("Label"):
        if label.lower() == "benign":
            continue

        positions = group.index.to_numpy()
        rate = (y_pred[positions] == 1).mean() * 100

        print(f"{label}: {rate:.2f}%")

    benign_mask = y_true == 0

    if benign_mask.any():
        false_positive_rate = (
            y_pred[benign_mask] == 1
        ).mean() * 100

        print(
            "\nBENIGN false-positive rate: "
            f"{false_positive_rate:.2f}%"
        )

    print(
        "\nNote: Web attacks were not included in training. "
        "This test measures generalization to a different attack "
        "category, not performance on live network traffic."
    )


if __name__ == "__main__":
    main()
