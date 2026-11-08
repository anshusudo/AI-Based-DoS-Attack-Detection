
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# Load the model trained on Friday's DDoS dataset
model = joblib.load("models/cicids2017_ddos_model.pkl")

# Load Wednesday's separate traffic capture
file_path = (
    "data/raw/cicids2017/MachineLearningCVE/"
    "Wednesday-workingHours.pcap_ISCX.csv"
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

print("Loading Wednesday traffic dataset...")

df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Map BENIGN to 0 and the selected DoS categories to 1.
dos_labels = [
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
]

allowed_labels = ["BENIGN"] + dos_labels
df = df[df["Label"].isin(allowed_labels)].copy()

print("\nActual label distribution:")
print(df["Label"].value_counts())

# Select a reproducible random sample
sample = df.sample(
    n=min(5000, len(df)),
    random_state=42,
).copy()

X = sample[FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Ground truth: normal = 0, selected DoS categories = 1
y_true = (sample["Label"] != "BENIGN").astype(int).to_numpy()

print(f"\nEvaluating {len(sample):,} records...")

# Predictions: BENIGN = 0, DDoS = 1
y_pred = model.predict(X)

print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))

print("\nConfusion Matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "DoS Attack"],
        zero_division=0,
    )
)

print(
    "\nNote: This is an external test on Wednesday's traffic. "
    "The model was trained on Friday's DDoS data, so this measures "
    "generalization to different DoS categories, not just DDoS detection."
)
