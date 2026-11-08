
import pandas as pd
from pathlib import Path

# Path to the real DDoS dataset
file_path = Path(
    "data/raw/cicids2017/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

print("Loading dataset...")

# Read the CSV
df = pd.read_csv(file_path, low_memory=False)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

# Inspect the actual class labels
print("\nLabel distribution:")
print(df["Label"].value_counts(dropna=False))

print("\nFirst five rows:")
print(df.head())

