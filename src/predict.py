
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/dos_model.pkl")

# Example traffic observations
new_traffic = pd.DataFrame([
    {
        "flow_duration": 125,
        "packets_per_second": 40,
        "bytes_per_second": 4100
    },
    {
        "flow_duration": 10,
        "packets_per_second": 950,
        "bytes_per_second": 85000
    }
])

# Make predictions
predictions = model.predict(new_traffic)

# Display results
for i, prediction in enumerate(predictions):
    print(f"\nTraffic sample {i + 1}")
    print("Prediction:", prediction)

    if prediction == "DOS":
        print("Interpretation: The model classified this sample as DOS.")
    else:
        print("Interpretation: The model classified this sample as BENIGN.")
