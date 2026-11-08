
from pathlib import Path
from datetime import datetime
import argparse

from scapy.all import sniff, wrpcap, IP

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "live" / "live_traffic.pcap"


def main():
    parser = argparse.ArgumentParser(
        description="Capture a short sample of network traffic."
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--interface",
        default="enp0s3",
        help="Network interface to capture from"
    )
    args = parser.parse_args()

    if args.seconds < 1 or args.seconds > 300:
        parser.error("--seconds must be between 1 and 300")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Interface: {args.interface}")
    print(f"Duration: {args.seconds} seconds")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("Capturing packets...")

    try:
        packets = sniff(
            iface=args.interface,
            filter="ip",
            timeout=args.seconds,
            store=True
        )
    except PermissionError:
        print("Permission denied. Run this script with sudo.")
        return
    except Exception as exc:
        print(f"Capture failed: {exc}")
        print("Check the interface name and packet-capture permissions.")
        return

    ipv4_packets = [packet for packet in packets if IP in packet]

    if not ipv4_packets:
        print("No IPv4 packets captured. Generate some normal network traffic and retry.")
        return

    wrpcap(str(OUTPUT_FILE), ipv4_packets)

    print(f"Packets captured: {len(ipv4_packets)}")
    print(f"Saved capture: {OUTPUT_FILE}")
    print("Next: run src/extract_live_features.py")


if __name__ == "__main__":
    main()
