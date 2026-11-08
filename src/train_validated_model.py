
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path("data/raw/cicids2017/MachineLearningCVE")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

DOS_LABELS = [
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
]

TRAIN_LIMIT_PER_LABEL = 30000
TEST_LIMIT_PER_LABEL = 5000


def load_and_split(filename):
    path = DATA_DIR / filename
    print(f"\nLoading {filename}...")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    allowed = ["BENIGN"] + DOS_LABELS
    df = df[df["Label"].isin(allowed)].copy()

    df = df[FEATURES + ["Label"]]

    # Split each original label BEFORE limiting the dataset.
    train_parts = []
    test_parts = []

    for label, group in df.groupby("Label"):
        train_group, test_group = train_test_split(
            group,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )

        train_parts.append(train_group)
        test_parts.append(test_group)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def limit_per_label(df, limit):
    parts = []

    for label, group in df.groupby("Label"):
        parts.append(
            group.sample(
                n=min(limit, len(group)),
                random_state=42,
            )
        )

    return pd.concat(parts, ignore_index=True)


def prepare_features(df):
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def main():
    train_parts = []
    test_parts = []

    for filename in FILES:
        train_df, test_df = load_and_split(filename)
        train_parts.append(train_df)
        test_parts.append(test_df)

    train_df = limit_per_label(
        pd.concat(train_parts, ignore_index=True),
        TRAIN_LIMIT_PER_LABEL,
    )

    test_df = limit_per_label(
        pd.concat(test_parts, ignore_index=True),
        TEST_LIMIT_PER_LABEL,
    )

    # Binary target: BENIGN = 0, all selected DoS types = 1.
    y_train = (train_df["Label"] != "BENIGN").astype(int)
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    print("\nTraining records:", len(train_df))
    print("Held-out test records:", len(test_df))

    print("\nTraining label distribution:")
    print(train_df["Label"].value_counts())

    print("\nHeld-out label distribution:")
    print(test_df["Label"].value_counts())

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("\nTraining validated DoS model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(
        "\nHeld-out accuracy:",
        f"{accuracy_score(y_test, predictions) * 100:.2f}%",
    )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    # Detection rates by actual attack category
    print("\nDetection rate by category:")

    for label in DOS_LABELS:
        mask = test_df["Label"].to_numpy() == label

        if mask.any():
            rate = (predictions[mask] == 1).mean() * 100
            print(f"{label}: {rate:.2f}%")

    output_path = MODEL_DIR / "cicids2017_validated_model.pkl"
    joblib.dump(model, output_path)

    print(f"\nModel saved to: {output_path}")
    print("Existing dashboard models have not been overwritten.")


if __name__ == "__main__":
    main()
