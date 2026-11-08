
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/raw/cicids2017/MachineLearningCVE"
MODEL_FILE = BASE_DIR / "models/cicids2017_expanded_model.pkl"

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
    "Tuesday-WorkingHours.pcap_ISCX.csv",
]

ATTACK_LABELS = {
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "FTP-Patator",
    "SSH-Patator",
}

SAMPLES_PER_CLASS = 500


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    samples = []

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Skipping missing file: {filename}")
            continue

        print(f"Reading {filename}...")
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        header.columns = header.columns.str.strip()

        required = FEATURES + ["Label"]
        missing = [c for c in required if c not in header.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")

        # Read only required columns, in chunks to limit memory use.
        selected = []
        # Normalize the CSV headers before selecting columns.
        for chunk in pd.read_csv(
            path,
            encoding="utf-8-sig",
            chunksize=50000,
            low_memory=False,
        ):
            chunk.columns = chunk.columns.str.strip()

            missing = [col for col in required if col not in chunk.columns]
            if missing:
                raise ValueError(
                    f"Missing columns in {filename}: {missing}"
                )

            chunk = chunk[required]
            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            benign = chunk[chunk["Label"] == "BENIGN"]
            attacks = chunk[chunk["Label"].isin(ATTACK_LABELS)]

            selected.extend(
                [benign.sample(min(len(benign), SAMPLES_PER_CLASS),
                               random_state=42)]
                if not benign.empty else []
            )
            selected.extend(
                [attacks.sample(min(len(attacks), SAMPLES_PER_CLASS),
                                random_state=42)]
                if not attacks.empty else []
            )

            # Stop once enough examples of each binary class have been collected.
            combined = pd.concat(selected, ignore_index=True)
            if (
                (combined["Label"] == "BENIGN").sum() >= SAMPLES_PER_CLASS
                and combined["Label"].isin(ATTACK_LABELS).sum()
                >= SAMPLES_PER_CLASS
            ):
                break

        if selected:
            samples.append(pd.concat(selected, ignore_index=True))

    if not samples:
        raise ValueError("No labeled samples were collected.")

    data = pd.concat(samples, ignore_index=True)
    data = data.drop_duplicates(subset=FEATURES + ["Label"])

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data["Label"].isin(ATTACK_LABELS).astype(int)

    valid = X.notna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    if y.nunique() < 2:
        raise ValueError("The sample must contain both benign and attack rows.")

    predictions = model.predict(X)

    print("\nEvaluation sample counts:")
    print(y.map({0: "BENIGN", 1: "ATTACK"}).value_counts().to_string())

    print(f"\nAccuracy: {accuracy_score(y, predictions) * 100:.2f}%")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    print(
        "\nNote: This is a diagnostic sample, not an independent benchmark. "
        "Use a held-out capture/session split for a trustworthy performance "
        "estimate. CICIDS2017 samples may overlap with model training data."
    )


if __name__ == "__main__":
    main()
