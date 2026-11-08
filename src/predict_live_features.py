
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models/cicids2017_expanded_model.pkl"
FEATURE_FILE = BASE_DIR / "data/live/live_features.csv"
OUTPUT_FILE = BASE_DIR / "data/live/live_predictions.csv"

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
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    data = pd.read_csv(FEATURE_FILE)

    missing = [column for column in FEATURES
               if column not in data.columns]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            "Features contain missing or invalid values."
        )

    print("Loading expanded Random Forest model...")
    model = joblib.load(MODEL_FILE)

    print(f"Analyzing {len(X)} extracted flows...")
    predictions = model.predict(X)

    # The training script encodes BENIGN as 0 and
    # every included attack category as 1.
    labels = ["ATTACK" if value == 1 else "BENIGN"
              for value in predictions]

    results = data.copy()
    results["Prediction"] = labels

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)

        if 1 in classes:
            attack_index = classes.index(1)
            results["Attack Probability (%)"] = (
                probabilities[:, attack_index] * 100
            ).round(2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nPreliminary prediction summary:")
    print(results["Prediction"].value_counts())

    print("\nFlow predictions:")
    print(results.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "\nWarning: These are experimental classifications. "
        "The simplified live features have not yet been "
        "validated against the training feature-generation "
        "pipeline. An ATTACK prediction is not proof of "
        "an active DoS/DDoS attack."
    )


if __name__ == "__main__":
    main()
