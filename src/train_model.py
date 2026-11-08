
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("data/network_traffic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate features (X) and labels (y)
X = df[[
    "flow_duration",
    "packets_per_second",
    "bytes_per_second"
]]

y = df["label"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Predict labels for the test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 2))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["BENIGN", "DOS"]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["BENIGN", "DOS"],
    zero_division=0
))

# 8. Save the trained model
joblib.dump(model, "models/dos_model.pkl")

print("\nModel saved to models/dos_model.pkl")
