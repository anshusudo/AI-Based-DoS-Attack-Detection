
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "evaluation_summary.csv"

results = [
    {
        "Test": "Wednesday DoS categories",
        "Records": 25000,
        "Metric": "Accuracy",
        "Value_Percent": 94.32,
        "Limitation": (
            "Same capture file was included in training"
        ),
    },
    {
        "Test": "Monday benign traffic",
        "Records": 10000,
        "Metric": "False-positive rate",
        "Value_Percent": 9.02,
        "Limitation": (
            "Benign-only sample; no attack recall measured"
        ),
    },
    {
        "Test": "Tuesday FTP/SSH attacks",
        "Records": 6000,
        "Metric": "Accuracy",
        "Value_Percent": 31.80,
        "Limitation": (
            "Attack categories outside the original DoS scope"
        ),
    },
    {
        "Test": "Thursday web attacks",
        "Records": 4180,
        "Metric": "Accuracy",
        "Value_Percent": 49.62,
        "Limitation": (
            "Web attacks outside the model's training scope"
        ),
    },
    {
        "Test": "Friday PortScan",
        "Records": 4000,
        "Metric": "Accuracy",
        "Value_Percent": 46.92,
        "Limitation": (
            "PortScan is not a DoS attack"
        ),
    },
]

df = pd.DataFrame(results)
df.to_csv(OUTPUT, index=False)

print("Evaluation report created:")
print(OUTPUT)
print()
print(df.to_string(index=False))
