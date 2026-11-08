from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/raw/cicids2017/MachineLearningCVE"
LIVE_FILE = BASE_DIR / "data/live/live_features.csv"

TRAIN_FILE = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

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


def clean_features(df):
    df.columns = df.columns.str.strip()

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    result = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def main():
    print("Loading training data...")
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    train = clean_features(train)

    print("Loading live features...")
    live = clean_features(pd.read_csv(LIVE_FILE))

    print("\nTraining feature statistics:")
    print(train.describe(percentiles=[0.5]).T[
        ["mean", "50%", "std", "min", "max"]
    ].to_string())

    print("\nLive feature statistics:")
    print(live.describe(percentiles=[0.5]).T[
        ["mean", "50%", "std", "min", "max"]
    ].to_string())

    print("\nTraining missing-value counts:")
    print(train.isna().sum().to_string())

    print("\nLive missing-value counts:")
    print(live.isna().sum().to_string())

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
