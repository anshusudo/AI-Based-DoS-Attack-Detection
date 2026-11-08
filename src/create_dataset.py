
import pandas as pd

# Educational sample network-traffic dataset
data = {
    "flow_duration": [
        120, 150, 100, 180, 140,
        10, 12, 8, 15, 9,
        130, 160, 110, 11, 14
    ],
    "packets_per_second": [
        40, 35, 45, 38, 42,
        900, 850, 1000, 950, 1100,
        36, 41, 39, 1050, 980
    ],
    "bytes_per_second": [
        4000, 3500, 4500, 3800, 4200,
        80000, 75000, 90000, 85000, 95000,
        3600, 4100, 3900, 92000, 87000
    ],
    "label": [
        "BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN",
        "DOS", "DOS", "DOS", "DOS", "DOS",
        "BENIGN", "BENIGN", "BENIGN", "DOS", "DOS"
    ]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Save the dataset
df.to_csv("data/network_traffic.csv", index=False)

# Display the dataset
print("Dataset created successfully!")
print(df)

# Display the number of samples in each class
print("\nClass distribution:")
print(df["label"].value_counts())
