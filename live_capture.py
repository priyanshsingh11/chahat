"""
SF-IDS Personal Traffic Evaluator & Live Capture
Records your own network browsing traffic (via live Wi-Fi sniffing or offline pcap),
extracts the 68 statistical flow features using cicflowmeter, and queries the running
FastAPI fog-node (/predict) to compute your real-world False Alarm Rate.

Usage:
    # 1. Live 10-minute browsing capture (requires sudo to sniff raw packets on macOS):
    sudo ./venv/bin/python live_capture.py --interface en0 --duration 600

    # 2. Or offline pcap evaluation:
    python live_capture.py --pcap my_browsing.pcap
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, Any, List

import requests

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Map the 68 CIC-IDS-2017 feature headers to cicflowmeter dictionary keys
MAPPING = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Header Length.1": "fwd_header_len",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min"
}


class FlowEvaluator:
    def __init__(self, api_url: str, meta_path: str, output_csv: str):
        self.api_url = api_url.rstrip("/")
        self.output_csv = output_csv
        with open(meta_path, "r") as f:
            self.features = json.load(f)["feature_names"]

        self.total = 0
        self.allowed = 0
        self.blocked = 0
        self.reviewed = 0
        self.latencies: List[float] = []

        # Open CSV output file
        self.csv_file = open(self.output_csv, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.features + ["Source IP", "Destination IP", "Verdict", "Confidence"])

    def process_flow_data(self, data: Dict[str, Any]):
        """Convert a single cicflowmeter flow dictionary to 68 features and send to /predict."""
        values = []
        for feat in self.features:
            k = MAPPING.get(feat)
            raw = data.get(k, 0.0) if k else 0.0
            try:
                val = float(raw)
                # Handle inf / nan if any
                if val != val or val == float("inf") or val == float("-inf"):
                    val = 0.0
            except Exception:
                val = 0.0
            values.append(val)

        src_ip = str(data.get("src_ip", "127.0.0.1"))
        dst_ip = str(data.get("dst_ip", "0.0.0.0"))
        dst_port = int(data.get("dst_port", 0) or 0)

        # Call FastAPI Fog Node /predict
        try:
            resp = requests.post(
                f"{self.api_url}/predict",
                json={"values": values, "src": src_ip},
                timeout=5
            )
            res = resp.json()
        except Exception as e:
            print(f"{RED}[API Error]{RESET} Could not connect to {self.api_url}/predict: {e}")
            return

        verdict = res.get("action", "UNKNOWN")
        pred_label = res.get("classification", "UNKNOWN")
        confidence = res.get("confidence", 0.0)
        latency = res.get("latency_ms", 0.0)

        self.total += 1
        if verdict == "ALLOW":
            self.allowed += 1
            color = GREEN
            tag = "✓ ALLOW "
        elif verdict == "BLOCK":
            self.blocked += 1
            color = RED
            tag = "✗ BLOCK "
        else:
            self.reviewed += 1
            color = YELLOW
            tag = "? REVIEW"

        self.latencies.append(latency)

        # Save to CSV
        self.csv_writer.writerow(values + [src_ip, dst_ip, verdict, confidence])
        self.csv_file.flush()

        # Log line to terminal
        print(f"{color}{tag}{RESET} pred={pred_label:<14} conf={confidence*100:5.1f}%  "
              f"{latency:4.1f}ms  {src_ip} -> {dst_ip}:{dst_port:<5}")

    def print_summary(self):
        self.csv_file.close()
        print("\n" + "=" * 65)
        print(f"{BOLD}         PERSONAL BROWSING EVALUATION REPORT{RESET}")
        print("=" * 65)
        if self.total == 0:
            print("No flows were completed during this session.")
            print("Tip: Browse a few websites (YouTube, Wikipedia, Google) to generate traffic.")
            return

        fpr = (self.blocked / self.total) * 100.0
        review_pct = (self.reviewed / self.total) * 100.0
        allow_pct = (self.allowed / self.total) * 100.0

        print(f"Total Flows Evaluated:        {BOLD}{self.total}{RESET}")
        print(f"Normal Traffic Allowed (OK):  {GREEN}{self.allowed}{RESET} ({allow_pct:.1f}%)")
        print(f"Flagged for Review:           {YELLOW}{self.reviewed}{RESET} ({review_pct:.1f}%)")
        print(f"False Alarms (Blocked):       {RED}{self.blocked}{RESET} ({fpr:.1f}%)")
        print("-" * 65)
        print(f"Real-World False Alarm Rate:  {BOLD}{RED if fpr > 2.0 else GREEN}{fpr:.2f}%{RESET}")
        if self.latencies:
            sorted_lat = sorted(self.latencies)
            median_lat = sorted_lat[len(sorted_lat) // 2]
            p95_lat = sorted_lat[int(len(sorted_lat) * 0.95) - 1]
            print(f"Median Inference Latency:     {CYAN}{median_lat:.2f} ms{RESET}")
            print(f"P95 Inference Latency:        {CYAN}{p95_lat:.2f} ms{RESET}")
        print("-" * 65)
        print(f"Recorded flows saved to:      {self.output_csv}")
        print("=" * 65 + "\n")


def evaluate_offline_pcap(pcap_path: str, evaluator: FlowEvaluator):
    from cicflowmeter.sniffer import create_sniffer
    print(f"\nProcessing pcap file: {pcap_path} ...")
    
    # Custom writer that feeds completed flows directly to evaluator
    class DirectWriter:
        def write(self, data):
            evaluator.process_flow_data(data)

    sniffer, session = create_sniffer(
        input_file=pcap_path,
        input_interface=None,
        output_mode=None,
        output=None,
        fields=None,
        verbose=False
    )
    session.output_writer = DirectWriter()
    sniffer.start()
    sniffer.join()
    session.flush_flows()
    evaluator.print_summary()


def evaluate_live_traffic(interface: str, duration_sec: int, evaluator: FlowEvaluator):
    from cicflowmeter.sniffer import create_sniffer
    print(f"\n{BOLD}Starting live traffic capture on interface: {CYAN}{interface}{RESET}")
    print(f"Evaluation duration: {duration_sec}s (Press {BOLD}Ctrl+C{RESET} anytime to stop and view results)")
    print(f"Browse normally now (Google, YouTube, Reddit, News, etc.)...\n")

    class DirectWriter:
        def write(self, data):
            evaluator.process_flow_data(data)

    sniffer, session = create_sniffer(
        input_file=None,
        input_interface=interface,
        output_mode=None,
        output=None,
        fields=None,
        verbose=False
    )
    session.output_writer = DirectWriter()

    def handle_exit(sig, frame):
        print(f"\nStopping capture and finalizing flows...")
        try:
            sniffer.stop()
        except Exception:
            pass
        session.flush_flows()
        evaluator.print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    sniffer.start()

    # Wait for the specified duration with progress countdown
    start_t = time.time()
    try:
        while time.time() - start_t < duration_sec:
            time.sleep(1)
    except KeyboardInterrupt:
        handle_exit(None, None)

    handle_exit(None, None)


def main():
    parser = argparse.ArgumentParser(description="SF-IDS Personal Traffic Evaluator")
    parser.add_argument("--interface", default="en0", help="Network interface to sniff (e.g. en0)")
    parser.add_argument("--duration", type=int, default=600, help="Live capture duration in seconds (default: 600 = 10 mins)")
    parser.add_argument("--pcap", default=None, help="Path to offline .pcap file to evaluate")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="FastAPI fog-node endpoint URL")
    parser.add_argument("--meta", default="artifacts/meta.json", help="Path to artifacts/meta.json")
    parser.add_argument("--output", default="personal_traffic.csv", help="Output CSV path for recorded flows")
    args = parser.parse_args()

    # Verify endpoint is up
    try:
        r = requests.get(f"{args.url}/health", timeout=3)
        if r.status_code != 200:
            sys.exit(f"Fog node at {args.url} returned status {r.status_code}. Is main.py running?")
    except Exception as e:
        sys.exit(f"Could not reach {args.url}. Please start your server first with: uvicorn main:app --port 8001")

    evaluator = FlowEvaluator(api_url=args.url, meta_path=args.meta, output_csv=args.output)

    if args.pcap:
        if not os.path.exists(args.pcap):
            sys.exit(f"File not found: {args.pcap}")
        evaluate_offline_pcap(args.pcap, evaluator)
    else:
        # Check root privilege for live packet capture
        if os.geteuid() != 0:
            print(f"{YELLOW}[Notice]{RESET} Capturing live packets on macOS requires root privileges.")
            print(f"Please run with {BOLD}sudo{RESET}:\n")
            print(f"    sudo ./venv/bin/python live_capture.py --interface {args.interface} --duration {args.duration}\n")
            sys.exit(1)
        evaluate_live_traffic(args.interface, args.duration, evaluator)


if __name__ == "__main__":
    main()
