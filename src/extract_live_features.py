
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from scapy.all import PcapReader, IP, TCP, UDP

BASE_DIR = Path(__file__).resolve().parent.parent
PCAP_FILE = BASE_DIR / "data/live/live_traffic.pcap"
OUTPUT_FILE = BASE_DIR / "data/live/live_features.csv"

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


def get_protocol(packet):
    if TCP in packet:
        return "TCP"
    if UDP in packet:
        return "UDP"
    return str(packet[IP].proto)


def extract_features():
    if not PCAP_FILE.exists():
        raise FileNotFoundError(
            f"Capture file not found: {PCAP_FILE}"
        )

    # Each bidirectional flow is identified by its IPs,
    # ports, and transport protocol.
    flows = {}

    print(f"Reading capture: {PCAP_FILE}")

    with PcapReader(str(PCAP_FILE)) as packets:
        for packet in packets:
            if IP not in packet:
                continue

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = get_protocol(packet)

            if TCP in packet:
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif UDP in packet:
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)
            else:
                src_port = 0
                dst_port = 0

            forward_key = (
                src_ip, dst_ip, src_port, dst_port, protocol
            )
            reverse_key = (
                dst_ip, src_ip, dst_port, src_port, protocol
            )

            if forward_key in flows:
                key = forward_key
                direction = "forward"
            elif reverse_key in flows:
                key = reverse_key
                direction = "backward"
            else:
                key = forward_key
                direction = "forward"
                flows[key] = {
                    "start": float(packet.time),
                    "end": float(packet.time),
                    "fwd_packets": 0,
                    "bwd_packets": 0,
                    "fwd_bytes": 0,
                    "bwd_bytes": 0,
                    "lengths": [],
                }

            flow = flows[key]
            timestamp = float(packet.time)
            length = len(packet)

            flow["start"] = min(flow["start"], timestamp)
            flow["end"] = max(flow["end"], timestamp)
            flow["lengths"].append(length)

            if direction == "forward":
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += length
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += length

    rows = []

    for flow in flows.values():
        duration_seconds = max(
            flow["end"] - flow["start"], 0.000001
        )

        total_packets = (
            flow["fwd_packets"] + flow["bwd_packets"]
        )
        total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]

        rows.append({
            "Flow Duration": duration_seconds * 1_000_000,
            "Total Fwd Packets": flow["fwd_packets"],
            "Total Backward Packets": flow["bwd_packets"],
            "Total Length of Fwd Packets": flow["fwd_bytes"],
            "Total Length of Bwd Packets": flow["bwd_bytes"],
            "Flow Bytes/s": total_bytes / duration_seconds,
            "Flow Packets/s": total_packets / duration_seconds,
            "Packet Length Mean": float(
                np.mean(flow["lengths"])
            ),
        })

    if not rows:
        raise ValueError(
            "No IPv4 traffic found in the capture."
        )

    result = pd.DataFrame(rows, columns=FEATURES)

    # Prevent invalid numerical values from entering the model.
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFlows extracted: {len(result)}")
    print(f"Feature columns: {len(result.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFirst five flows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    extract_features()
