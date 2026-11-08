
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the real dataset
file_path = (
    "data/raw/cicids2017/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

print("Loading real CIC-IDS2017 data...")
df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()

# 2. Select useful traffic features
features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean"
]

# 3. Clean labels and select BENIGN/DDoS records
df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["Label"].map({"BENIGN": 0, "DDoS": 1})

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts().rename(index={0: "BENIGN", 1: "DDoS"}))

# 4. Split before learning preprocessing statistics
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Build a preprocessing + classification pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train the model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 7. Evaluate on held-out records
predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nConfusion Matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    zero_division=0
))

# 8. Save the complete pipeline
joblib.dump(model, "models/cicids2017_ddos_model.pkl")

print("\nReal-data model saved successfully!")
print("File: models/cicids2017_ddos_model.pkl")
