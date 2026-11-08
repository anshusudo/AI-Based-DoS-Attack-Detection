
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "evaluation_summary.csv"

st.title("Model Evaluation Report")

st.write(
    "Review the model's evaluation results across different "
    "CIC-IDS2017 traffic captures."
)

if not REPORT_PATH.exists():
    st.error(
        "Evaluation report not found. Run "
        "python src/create_evaluation_report.py first."
    )
    st.stop()

df = pd.read_csv(REPORT_PATH)

st.subheader("Evaluation Results")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Visual Comparison")

accuracy_rows = df[df["Metric"] == "Accuracy"]

if not accuracy_rows.empty:
    chart_data = accuracy_rows.set_index("Test")[
        "Value_Percent"
    ]
    st.bar_chart(chart_data)

st.subheader("False-Positive Rate")

fpr_rows = df[df["Metric"] == "False-positive rate"]

if not fpr_rows.empty:
    st.metric(
        "Monday benign traffic",
        f"{fpr_rows.iloc[0]['Value_Percent']:.2f}%"
    )

st.warning(
    "These tests use different traffic samples and evaluation "
    "methods. They are not directly comparable benchmarks. "
    "Accuracy on an attack dataset does not establish "
    "performance on live traffic."
)
