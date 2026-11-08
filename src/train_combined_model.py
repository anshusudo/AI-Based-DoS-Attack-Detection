
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

ATTACK_LABELS = [
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
]

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

MAX_PER_LABEL_PER_FILE = 30000


def load_file(filename):
    path = DATA_DIR / filename
    print(f"\nLoading: {filename}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    required = FEATURES + ["Label"]
    df = df[required].copy()

    # Keep benign and the selected attack categories only.
    allowed = ["BENIGN"] + ATTACK_LABELS
    df = df[df["Label"].isin(allowed)].copy()

    # Limit each label so training remains manageable.
    parts = []

    for label, group in df.groupby("Label"):
        group = group.sample(
            n=min(MAX_PER_LABEL_PER_FILE, len(group)),
            random_state=42,
        )
        parts.append(group)

    result = pd.concat(parts, ignore_index=True)

    # Binary target: normal traffic = 0, attack traffic = 1.
    result["Target"] = (
        result["Label"] != "BENIGN"
    ).astype(int)

    print("Selected labels:")
    print(result["Label"].value_counts().to_string())

    return result


def main():
    frames = [load_file(filename) for filename in FILES]
    data = pd.concat(frames, ignore_index=True)

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data["Target"]

    print("\nCombined dataset size:", len(data))
    print("\nBinary class distribution:")
    print(y.map({0: "BENIGN", 1: "ATTACK"}).value_counts())

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
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("\nTraining combined attack-detection model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(
        "\nTest accuracy:",
        round(accuracy_score(y_test, predictions), 4),
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

    output_path = MODEL_DIR / "cicids2017_attack_model.pkl"
    joblib.dump(model, output_path)

    print(f"\nNew model saved to: {output_path}")
    print("The original model has not been overwritten.")


if __name__ == "__main__":
    main()
