
import ipaddress
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based DoS Attack Detection",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cicids2017_validated_model.pkl"

LIVE_DIR = BASE_DIR / "data" / "live"
CAPTURE_FILE = LIVE_DIR / "live_traffic.pcap"
FEATURES_FILE = LIVE_DIR / "live_features.csv"
PREDICTIONS_FILE = LIVE_DIR / "live_predictions.csv"

CAPTURE_SCRIPT = BASE_DIR / "src" / "capture_live_traffic.py"
FEATURE_SCRIPT = BASE_DIR / "src" / "extract_live_features.py"
PREDICTION_SCRIPT = BASE_DIR / "src" / "predict_live_features.py"

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

ATTACK_LABELS = {
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
}

MAX_ROWS = 5000


# =========================================================
# 2. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(model_path, model_mtime):
    return joblib.load(model_path)


st.title("🛡️ AI-Based DoS Attack Detection System")

st.write(
    "Check website reachability, classify network traffic, "
    "and inspect live traffic predictions using a Random Forest "
    "model trained on CIC-IDS2017."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found.\n\n"
        "Expected: models/cicids2017_validated_model.pkl"
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# =========================================================
# 3. URL VALIDATION
# =========================================================

def validate_public_url(url):
    """Basic validation before making an HTTP request."""

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https"):
            return False, "Use a URL beginning with http:// or https://."

        if not parsed.hostname:
            return False, "Enter a valid website hostname."

        if parsed.username or parsed.password:
            return False, "URLs containing embedded credentials are blocked."

        port = parsed.port or (
            443 if parsed.scheme == "https" else 80
        )

        addresses = socket.getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )

        if not addresses:
            return False, "The hostname could not be resolved."

        for address in addresses:
            ip = ipaddress.ip_address(
                address[4][0].split("%")[0]
            )

            if not ip.is_global:
                return False, "Private or non-public destinations are blocked."

        return True, "URL validated."

    except (ValueError, OSError):
        return False, "The URL or hostname could not be validated."


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def run_command(command, timeout_seconds):
    """Run a project command and return its output."""

    return subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def run_live_pipeline(interface, duration):
    """Capture traffic, extract features, and generate predictions."""

    python_exe = str(BASE_DIR / ".venv" / "bin" / "python")

    if not Path(python_exe).exists():
        raise RuntimeError(
            "Project virtual environment not found at .venv/bin/python."
        )

    scripts = [
        (
            "Packet capture",
            [
                python_exe,
                str(CAPTURE_SCRIPT),
                "--seconds", str(duration),
                "--interface", interface,
            ],
            duration + 60,
        ),
        (
            "Feature extraction",
            [python_exe, str(FEATURE_SCRIPT)],
            120,
        ),
        (
            "AI prediction",
            [python_exe, str(PREDICTION_SCRIPT)],
            120,
        ),
    ]

    for stage, command, timeout_seconds in scripts:

        if not Path(command[1]).exists():
            raise RuntimeError(
                f"{stage} script not found: {command[1]}"
            )

        with st.status(
            f"Running {stage}...",
            expanded=True,
        ) as status:

            try:
                result = run_command(command, timeout_seconds)

            except subprocess.TimeoutExpired:
                status.update(
                    label=f"{stage} timed out",
                    state="error",
                )
                raise RuntimeError(
                    f"{stage} exceeded its {timeout_seconds}-second timeout."
                )

            if result.stdout.strip():
                st.code(result.stdout)

            if result.returncode != 0:
                if result.stderr.strip():
                    st.code(result.stderr)

                status.update(
                    label=f"{stage} failed",
                    state="error",
                )

                raise RuntimeError(
                    f"{stage} failed with exit code {result.returncode}. "
                    "Check the output above."
                )

            status.update(
                label=f"{stage} completed",
                state="complete",
            )

    if not PREDICTIONS_FILE.exists():
        raise RuntimeError(
            "The prediction script completed, but "
            "data/live/live_predictions.csv was not found."
        )


def show_file_status(description, file_path):
    """Display whether a pipeline output file exists."""

    if file_path.exists():
        modified = pd.Timestamp(
            file_path.stat().st_mtime,
            unit="s",
        ).strftime("%Y-%m-%d %H:%M:%S")

        st.success(f"{description}: Available")
        st.caption(f"Last modified: {modified}")
    else:
        st.warning(f"{description}: Not available")


# =========================================================
# 5. DASHBOARD TABS
# =========================================================

tab_website, tab_traffic, tab_live = st.tabs(
    [
        "🌐 Website Check",
        "🤖 AI Traffic Analysis",
        "📡 Live Traffic Monitoring",
    ]
)


# =========================================================
# 6. WEBSITE REACHABILITY CHECK
# =========================================================

with tab_website:

    st.header("Website Reachability Check")

    st.info(
        "Use only websites you own or are authorized to monitor. "
        "One HTTP request cannot establish whether a website is "
        "experiencing a DoS attack."
    )

    url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        key="website_url",
    )

    if st.button("Check Website", type="primary"):

        if not url.strip():
            st.warning("Please enter a website URL.")

        else:
            valid, message = validate_public_url(url)

            if not valid:
                st.error(message)

            else:
                try:
                    with st.spinner("Checking website reachability..."):
                        response = requests.get(
                            url.strip(),
                            timeout=8,
                            allow_redirects=False,
                            stream=True,
                            headers={
                                "User-Agent": (
                                    "Authorized-Website-Monitor/1.0"
                                )
                            },
                        )

                        status_code = response.status_code
                        response_time = (
                            response.elapsed.total_seconds() * 1000
                        )
                        response.close()

                    col1, col2 = st.columns(2)

                    col1.metric("HTTP Status Code", status_code)
                    col2.metric(
                        "Response Time",
                        f"{response_time:.2f} ms",
                    )

                    if 200 <= status_code < 300:
                        st.success("The website responded successfully.")
                    elif 300 <= status_code < 400:
                        st.warning(
                            "The website returned a redirect. "
                            "Redirects are not followed."
                        )
                    else:
                        st.warning(
                            "The website returned an HTTP error. "
                            "This alone does not indicate a DoS attack."
                        )

                except requests.RequestException as exc:
                    st.error(f"The request failed: {exc}")


# =========================================================
# 7. AI NETWORK TRAFFIC ANALYSIS
# =========================================================

with tab_traffic:

    st.header("AI-Based Network Traffic Analysis")

    st.write(
        "Upload a network traffic CSV to classify records as "
        "BENIGN or ATTACK."
    )

    st.info(
        "The model uses eight numerical features. "
        "Up to 5,000 records are sampled per upload."
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        key="traffic_csv",
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(
                uploaded_file,
                low_memory=False,
            )

            data.columns = data.columns.str.strip()

            st.metric("Uploaded Records", f"{len(data):,}")

            if data.empty:
                st.warning("The uploaded CSV contains no records.")

            else:
                missing_features = [
                    feature
                    for feature in FEATURES
                    if feature not in data.columns
                ]

                if missing_features:
                    st.error(
                        "Missing required features: "
                        + ", ".join(missing_features)
                    )

                else:
                    sample = data.sample(
                        n=min(MAX_ROWS, len(data)),
                        random_state=42,
                    ).copy()

                    X = sample[FEATURES].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    X = X.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )

                    if X.isna().any().any():
                        invalid_counts = X.isna().sum()
                        invalid_counts = invalid_counts[
                            invalid_counts > 0
                        ]

                        st.warning(
                            "Invalid or missing feature values were found. "
                            "The model's pipeline may impute these values "
                            "if configured to do so."
                        )

                        st.dataframe(
                            invalid_counts.rename(
                                "Invalid values"
                            ).to_frame()
                        )

                    with st.spinner("Classifying traffic..."):
                        predictions = model.predict(X)

                    predicted_labels = np.where(
                        predictions == 1,
                        "ATTACK",
                        "BENIGN",
                    )

                    results = sample[FEATURES].copy()
                    results["Prediction"] = predicted_labels

                    st.subheader("Prediction Summary")

                    counts = pd.Series(
                        predicted_labels
                    ).value_counts()

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "BENIGN Records",
                        f"{int(counts.get('BENIGN', 0)):,}",
                    )
                    col2.metric(
                        "ATTACK Records",
                        f"{int(counts.get('ATTACK', 0)):,}",
                    )

                    if "Label" in sample.columns:

                        actual_labels = (
                            sample["Label"]
                            .astype(str)
                            .str.strip()
                        )

                        actual_binary = actual_labels.map(
                            lambda label: (
                                0 if label == "BENIGN"
                                else 1 if label in ATTACK_LABELS
                                else np.nan
                            )
                        )

                        valid = actual_binary.notna()

                        if valid.any():

                            y_true = (
                                actual_binary[valid]
                                .astype(int)
                                .to_numpy()
                            )

                            y_pred = predictions[valid.to_numpy()]

                            st.subheader(
                                "Evaluation Against Actual Labels"
                            )

                            st.metric(
                                "Accuracy on Recognized Labels",
                                f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
                            )

                            cm = confusion_matrix(
                                y_true,
                                y_pred,
                                labels=[0, 1],
                            )

                            st.write(
                                "Confusion Matrix "
                                "(rows = actual, columns = predicted)"
                            )

                            st.dataframe(
                                pd.DataFrame(
                                    cm,
                                    index=[
                                        "Actual BENIGN",
                                        "Actual ATTACK",
                                    ],
                                    columns=[
                                        "Predicted BENIGN",
                                        "Predicted ATTACK",
                                    ],
                                ),
                                use_container_width=True,
                            )

                            report = classification_report(
                                y_true,
                                y_pred,
                                labels=[0, 1],
                                target_names=["BENIGN", "ATTACK"],
                                output_dict=True,
                                zero_division=0,
                            )

                            st.write("Precision, Recall and F1-Score")
                            st.dataframe(
                                pd.DataFrame(report).T.round(4),
                                use_container_width=True,
                            )

                            if not valid.all():
                                st.warning(
                                    f"{int((~valid).sum())} records with "
                                    "unrecognized labels were excluded "
                                    "from evaluation."
                                )

                            st.warning(
                                "These metrics describe the uploaded sample. "
                                "They are not independent validation if the "
                                "records were used during training."
                            )

                        else:
                            st.info(
                                "No recognized ground-truth labels were "
                                "found. Evaluation metrics are unavailable."
                            )

                    else:
                        st.info(
                            "No Label column was found. "
                            "Predictions are shown without evaluation metrics."
                        )

                    st.subheader("Traffic Predictions")

                    st.dataframe(
                        results,
                        use_container_width=True,
                        height=400,
                    )

                    st.download_button(
                        "Download Prediction Results",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="dos_detection_results.csv",
                        mime="text/csv",
                    )

                    st.warning(
                        "The model predicts BENIGN or ATTACK based on "
                        "patterns learned from selected CIC-IDS2017 "
                        "categories. Predictions are not proof of an "
                        "active attack."
                    )

        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")


# =========================================================
# 8. LIVE NETWORK TRAFFIC MONITORING
# =========================================================

with tab_live:

    st.header("Live Network Traffic Monitoring")

    st.write(
        "Capture traffic on your Ubuntu machine, extract flow "
        "features, and generate fresh model predictions."
    )

    st.info(
        "Capture only traffic you are authorized to monitor. "
        "The live detector is experimental until its features "
        "are validated against the training pipeline."
    )

    st.subheader("Capture Settings")

    col1, col2 = st.columns(2)

    with col1:
        interface = st.selectbox(
            "Network interface",
            options=["enp0s3", "lo", "any"],
            index=0,
            key="live_interface",
        )

    with col2:
        duration = st.selectbox(
            "Capture duration",
            options=[10, 30, 45, 60],
            index=1,
            format_func=lambda seconds: f"{seconds} seconds",
            key="live_duration",
        )

    col1, col2 = st.columns(2)

    with col1:
        capture_button = st.button(
            "▶ Capture Fresh Traffic & Analyze",
            type="primary",
            key="capture_and_analyze",
            use_container_width=True,
        )

    with col2:
        refresh_button = st.button(
            "🔄 Refresh Existing Results",
            key="refresh_live",
            use_container_width=True,
        )

    if capture_button:

        missing_scripts = [
            str(script)
            for script in [
                CAPTURE_SCRIPT,
                FEATURE_SCRIPT,
                PREDICTION_SCRIPT,
            ]
            if not script.exists()
        ]

        if missing_scripts:
            st.error(
                "Required scripts were not found:\n\n"
                + "\n".join(missing_scripts)
            )

        else:
            try:
                with st.spinner(
                    f"Capturing traffic on {interface} for "
                    f"{duration} seconds..."
                ):
                    run_live_pipeline(interface, duration)

                st.success(
                    "Capture, feature extraction, and prediction "
                    "completed successfully."
                )

            except Exception as exc:
                st.error(f"Live pipeline failed: {exc}")

    # -----------------------------------------
    # Pipeline status
    # -----------------------------------------

    st.divider()
    st.subheader("Live Pipeline Status")

    status_items = [
        ("Packet Capture", CAPTURE_FILE),
        ("Extracted Features", FEATURES_FILE),
        ("AI Predictions", PREDICTIONS_FILE),
    ]

    status_columns = st.columns(3)

    for column, (description, file_path) in zip(
        status_columns,
        status_items,
    ):
        with column:
            show_file_status(description, file_path)

    # -----------------------------------------
    # Read the current prediction file
    # -----------------------------------------

    if refresh_button or capture_button:

        if not PREDICTIONS_FILE.exists():
            st.warning(
                "No predictions file exists yet. Capture and process "
                "traffic first."
            )

        else:
            try:
                live_results = pd.read_csv(
                    PREDICTIONS_FILE,
                    low_memory=False,
                )

                live_results.columns = (
                    live_results.columns.str.strip()
                )

                if live_results.empty:
                    st.warning(
                        "The prediction file contains no records."
                    )

                else:
                    st.success(
                        "Loaded the latest prediction file."
                    )

                    prediction_column = next(
                        (
                            column
                            for column in live_results.columns
                            if column.lower() == "prediction"
                        ),
                        None,
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Flows",
                        f"{len(live_results):,}",
                    )

                    if prediction_column is not None:

                        labels = (
                            live_results[prediction_column]
                            .astype(str)
                            .str.strip()
                            .str.upper()
                        )

                        benign_count = int(
                            (labels == "BENIGN").sum()
                        )
                        attack_count = int(
                            (labels == "ATTACK").sum()
                        )

                        col2.metric(
                            "BENIGN Flows",
                            f"{benign_count:,}",
                        )
                        col3.metric(
                            "ATTACK Flows",
                            f"{attack_count:,}",
                        )

                        if attack_count > 0:
                            st.warning(
                                f"The model flagged {attack_count} flow(s) "
                                "as ATTACK. Investigate and validate "
                                "the traffic before drawing conclusions."
                            )
                        else:
                            st.success(
                                "No flows were classified as ATTACK "
                                "in these results."
                            )

                    else:
                        col2.metric("BENIGN Flows", "N/A")
                        col3.metric("ATTACK Flows", "N/A")

                        st.warning(
                            "No Prediction column was found in the CSV."
                        )

                    probability_column = next(
                        (
                            column
                            for column in live_results.columns
                            if column.lower() in {
                                "attack probability (%)",
                                "attack probability",
                            }
                        ),
                        None,
                    )

                    if probability_column is not None:
                        probabilities = pd.to_numeric(
                            live_results[probability_column],
                            errors="coerce",
                        ).dropna()

                        if not probabilities.empty:
                            st.metric(
                                "Highest Reported Attack Probability",
                                f"{probabilities.max():.2f}%",
                            )

                    st.subheader("Latest Flow Predictions")

                    st.dataframe(
                        live_results,
                        use_container_width=True,
                        height=400,
                    )

                    st.download_button(
                        "Download Live Predictions",
                        data=live_results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="live_predictions.csv",
                        mime="text/csv",
                        key="download_live_predictions",
                    )

                    st.caption(
                        "The results correspond to the latest completed "
                        "prediction run. Similar traffic can produce "
                        "similar predictions."
                    )

                    st.warning(
                        "Live predictions remain experimental. The live "
                        "feature extractor must be validated against the "
                        "model's training feature definitions before "
                        "these predictions can be treated as reliable "
                        "DoS detection."
                    )

            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read live predictions: {exc}")

            except Exception as exc:
                st.error(
                    f"Unexpected error loading live results: {exc}"
                )

    st.caption(
        "Capture Fresh Traffic & Analyze runs the three existing "
        "scripts. Refresh Existing Results only reloads the CSV."
    )
