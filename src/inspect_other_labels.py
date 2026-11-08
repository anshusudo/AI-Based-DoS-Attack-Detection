import pandas as pd
from pathlib import Path

data_dir = Path("data/raw/cicids2017/MachineLearningCVE")

files = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]

for filename in files:
    path = data_dir / filename

    if not path.exists():
        print(f"\nFile not found: {filename}")
        continue

    print(f"\n--- {filename} ---")

    try:
        labels = pd.read_csv(
            path,
            usecols=lambda column: column.strip() == "Label",
            low_memory=False,
        )

        labels.columns = labels.columns.str.strip()
        labels["Label"] = labels["Label"].astype(str).str.strip()

        print(labels["Label"].value_counts().to_string())

    except Exception as error:
        print(f"Could not inspect file: {error}")
