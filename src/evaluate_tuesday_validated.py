
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
    / "Tuesday-WorkingHours.pcap_ISCX.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models/cicids2017_validated_model.pkl"
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

ATTACK_LABELS = ["FTP-Patator", "SSH-Patator"]
SAMPLE_PER_LABEL = 2000


def main():
    print("Loading validated model...")

    model = joblib.load(MODEL_PATH)

    print("Loading Tuesday traffic dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    required = FEATURES + ["Label"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\nOriginal label distribution:")
    print(df["Label"].value_counts())

    # Select a reproducible sample from each category.
    selected = []

    for label in ["BENIGN"] + ATTACK_LABELS:
        group = df[df["Label"] == label]

        if group.empty:
            print(f"Warning: label not found: {label}")
            continue

        count = min(SAMPLE_PER_LABEL, len(group))

        selected.append(
            group.sample(n=count, random_state=42)
        )

    if not selected:
        raise ValueError("No recognized traffic records were found.")

    test_df = pd.concat(selected, ignore_index=True)

    X = test_df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    # The model's training labels were:
    # BENIGN = 0; selected DoS/DDoS categories = 1.
    # For this external test, benign is 0 and the two
    # unseen attack categories are treated as attack = 1.
    y_true = (
        test_df["Label"] != "BENIGN"
    ).astype(int).to_numpy()

    print(f"\nRecords evaluated: {len(test_df):,}")
    print("\nEvaluation label distribution:")
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
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    print("\nDetection rate by actual category:")

    for label in ATTACK_LABELS:
        mask = test_df["Label"].to_numpy() == label

        if mask.any():
            rate = (y_pred[mask] == 1).mean() * 100
            print(f"{label}: {rate:.2f}%")

    benign_mask = test_df["Label"].to_numpy() == "BENIGN"

    if benign_mask.any():
        false_positive_rate = (
            y_pred[benign_mask] == 1
        ).mean() * 100

        print(
            "BENIGN false-positive rate: "
            f"{false_positive_rate:.2f}%"
        )

    print(
        "\nNote: FTP-Patator and SSH-Patator are brute-force "
        "attacks, not the DoS/DDoS categories used to train "
        "this model. This is an out-of-distribution test, "
        "not a direct measurement of DoS detection performance."
    )


if __name__ == "__main__":
    main()
