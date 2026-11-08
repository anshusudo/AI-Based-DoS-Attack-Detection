
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

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/raw/cicids2017/MachineLearningCVE"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
]

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

ATTACK_LABELS = [
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "FTP-Patator",
    "SSH-Patator",
]

SAMPLE_PER_LABEL = 20000


def load_dataset(filename):
    path = DATA_DIR / filename
    print(f"\nLoading {filename}...")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    missing = [
        col for col in FEATURES + ["Label"]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    allowed = ["BENIGN"] + ATTACK_LABELS
    df = df[df["Label"].isin(allowed)].copy()

    # Convert the selected features to numeric values.
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    df[FEATURES] = X
    return df[FEATURES + ["Label"]]


def sample_per_label(df):
    samples = []

    for label, group in df.groupby("Label"):
        n = min(SAMPLE_PER_LABEL, len(group))

        samples.append(
            group.sample(n=n, random_state=42)
        )

    return pd.concat(samples, ignore_index=True)


def main():
    # Load and combine the selected traffic categories.
    datasets = []

    for filename in FILES:
        datasets.append(load_dataset(filename))

    data = pd.concat(datasets, ignore_index=True)

    # Remove duplicate feature/label rows to reduce repeated records.
    data = data.drop_duplicates()

    print("\nAvailable labels:")
    print(data["Label"].value_counts())

    # Limit the influence of very large categories.
    data = sample_per_label(data)

    print("\nSampled label distribution:")
    print(data["Label"].value_counts())

    # Binary classification target.
    y = (data["Label"] != "BENIGN").astype(int)
    X = data[FEATURES]

    # Split before training. This is a record-level split;
    # it does not guarantee separation by capture session.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("\nTraining expanded attack-detection model...")
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

    print("\nAttack detection rate by category:")

    for label in ATTACK_LABELS:
        mask = data.loc[X_test.index, "Label"].to_numpy() == label

        if mask.any():
            rate = (predictions[mask] == 1).mean() * 100
            print(f"{label}: {rate:.2f}%")

    output = MODEL_DIR / "cicids2017_expanded_model.pkl"
    joblib.dump(model, output)

    print(f"\nExpanded model saved to: {output}")
    print("The existing dashboard model has not been overwritten.")


if __name__ == "__main__":
    main()
