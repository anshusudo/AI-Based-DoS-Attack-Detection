
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# Load the improved model
model = joblib.load("models/cicids2017_attack_model.pkl")

# Tuesday's traffic was not used to train this model
file_path = (
    "data/raw/cicids2017/MachineLearningCVE/"
    "Tuesday-WorkingHours.pcap_ISCX.csv"
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
MAX_PER_LABEL = 2000

print("Loading Tuesday traffic dataset...")

df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

print("\nOriginal label distribution:")
print(df["Label"].value_counts().to_string())

# Select BENIGN and the two unseen attack categories
allowed = ["BENIGN"] + ATTACK_LABELS
df = df[df["Label"].isin(allowed)].copy()

# Balanced-by-category sample for a clearer comparison
samples = []

for label in allowed:
    group = df[df["Label"] == label]

    if not group.empty:
        samples.append(
            group.sample(
                n=min(MAX_PER_LABEL, len(group)),
                random_state=42,
            )
        )

sample = pd.concat(samples, ignore_index=True)

# Prepare the eight features expected by the model
X = sample[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Actual labels: BENIGN = 0, attack = 1
y_true = (
    sample["Label"] != "BENIGN"
).astype(int).to_numpy()

print("\nEvaluation sample distribution:")
print(sample["Label"].value_counts().to_string())
print(f"\nTotal records evaluated: {len(sample):,}")

print("\nRunning predictions...")
y_pred = model.predict(X)

print(
    "\nAccuracy:",
    f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
)

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

# Examine results separately for each original attack category
print("\nDetection results by actual traffic category:")

sample = sample.copy()
sample["Actual"] = sample["Label"]
sample["Predicted"] = np.where(
    y_pred == 1,
    "ATTACK",
    "BENIGN",
)

for label in allowed:
    group = sample[sample["Actual"] == label]

    if not group.empty:
        detected = (group["Predicted"] == "ATTACK").sum()
        total = len(group)

        print(
            f"{label}: {detected}/{total} classified as ATTACK "
            f"({detected / total * 100:.2f}%)"
        )

print(
    "\nNote: These are attack categories not used during training. "
    "Results measure generalization on this dataset, not performance "
    "on live network traffic."
)
