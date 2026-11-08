from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/raw/cicids2017/MachineLearningCVE"

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

parts = []

for filename in FILES:
    path = DATA_DIR / filename
    print(f"Reading {filename}...")

    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = FEATURES + ["Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    df = df[required].copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].replace([float("inf"), -float("inf")], float("nan"))
    df = df.dropna(subset=FEATURES)

    # Keep only labels used by the expanded model.
    allowed = {
        "BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye",
        "DoS slowloris", "DoS Slowhttptest",
        "FTP-Patator", "SSH-Patator",
    }
    df = df[df["Label"].isin(allowed)].copy()
    df["SourceFile"] = filename
    parts.append(df)

data = pd.concat(parts, ignore_index=True)

print("\nRows after cleaning:", len(data))
print("Rows with duplicate feature values, regardless of label:",
      data.duplicated(subset=FEATURES, keep=False).sum())
print("Exact duplicate feature-and-label rows:",
      data.duplicated(subset=FEATURES + ["Label"], keep=False).sum())

print("\nRows by source file:")
print(data["SourceFile"].value_counts().to_string())

print("\nLabel distribution:")
print(data["Label"].value_counts().to_string())
