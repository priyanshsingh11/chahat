"""
SF-IDS - Level 1: Cross-Dataset Evaluation on CSE-CIC-IDS2018
Evaluates the DCNN-BiLSTM model (trained on CIC-IDS-2017) on unseen network traffic
recorded in 2018 on an entirely different AWS enterprise network environment.

Maps shortened 2018 column names to 2017 features, standardizes labels,
runs inference, and produces a comparative generalization report.

Usage:
    python evaluate_2018.py
    python evaluate_2018.py --csv data/raw/Thursday-22-02-2018_WebAttacks.csv
"""

import argparse
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from model import DCNNBiLSTM

# ANSI Color formatting
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# 2018 shortened column names -> 2017 full column names
MAPPING_2018_TO_2017 = {
    "Dst Port": "Destination Port",
    "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min"
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SF-IDS model on CSE-CIC-IDS2018")
    parser.add_argument("--csv", default="data/raw/Thursday-22-02-2018_WebAttacks.csv",
                        help="Path to 2018 CSV file")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size for inference")
    parser.add_argument("--art", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}=== SF-IDS Level 1: Cross-Dataset Evaluation on CSE-CIC-IDS2018 ==={RESET}\n")

    # 1. Load trained artifacts
    meta = json.load(open(f"{args.art}/meta.json"))
    expected_features = meta["feature_names"]
    classes = meta["classes"]
    scaler = joblib.load(f"{args.art}/scaler.pkl")
    encoder = joblib.load(f"{args.art}/encoder.pkl")
    ckpt = torch.load(f"{args.art}/sf_ids_model.pth", map_location="cpu")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DCNNBiLSTM(ckpt["num_features"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loaded trained model: {len(expected_features)} features | classes: {classes}")
    print(f"Reading 2018 dataset: {args.csv} ...")

    # 2. Read 2018 dataset
    df = pd.read_csv(args.csv, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    total_raw_rows = len(df)
    print(f"Loaded {total_raw_rows:,} raw flow records.")

    df.dropna(subset=["Label"], inplace=True)

    # In 2018: "Benign" -> "BENIGN"; "Brute Force -Web", "Brute Force -XSS", "SQL Injection" -> "Web Attack"
    def normalize_label(l) -> str:
        l_str = str(l).strip()
        l_lower = l_str.lower()
        if "benign" in l_lower:
            return "BENIGN"
        if any(w in l_lower for w in ["brute force", "xss", "sql injection", "web"]):
            return "Web Attack"
        return l_str

    df["Label"] = df["Label"].apply(normalize_label)
    # Keep only the classes the model was trained on
    df = df[df["Label"].isin(classes)].copy()

    print("\n2018 Class Distribution:")
    print(df["Label"].value_counts().to_string())

    # 4. Map columns to 2017 features
    mapped_df = pd.DataFrame()
    for col_2018, col_2017 in MAPPING_2018_TO_2017.items():
        if col_2018 in df.columns:
            mapped_df[col_2017] = pd.to_numeric(df[col_2018], errors="coerce")
        else:
            mapped_df[col_2017] = 0.0

    # Handle duplicate/special column: "Fwd Header Length.1" was identical to "Fwd Header Length" in 2017
    if "Fwd Header Length.1" in expected_features and "Fwd Header Length" in mapped_df.columns:
        mapped_df["Fwd Header Length.1"] = mapped_df["Fwd Header Length"]

    # Ensure all expected features are present in the exact order
    for feat in expected_features:
        if feat not in mapped_df.columns:
            mapped_df[feat] = 0.0

    X_2018 = mapped_df[expected_features].copy()
    X_2018.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_2018.fillna(0.0, inplace=True)

    y_2018 = encoder.transform(df["Label"].values)

    # 5. Standard scale using 2017 scaler
    X_scaled = scaler.transform(X_2018.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_2018, dtype=torch.long)

    # 6. Run Inference
    print(f"\nRunning DCNN-BiLSTM inference on {len(X_tensor):,} flows ({device}) ...")
    t0 = time.time()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), args.batch):
            batch_x = X_tensor[i:i + args.batch].to(device)
            out = model(batch_x)
            preds.extend(out.argmax(dim=1).cpu().numpy())
    total_sec = time.time() - t0
    y_pred = np.array(preds)
    latency_ms = (total_sec / len(X_tensor)) * 1000

    # 7. Metrics & Classification Report
    acc = accuracy_score(y_2018, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_2018, y_pred, average="macro", zero_division=0)
    report = classification_report(y_2018, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_2018, y_pred, labels=range(len(classes)))

    print("\n" + "=" * 65)
    print(f"{BOLD}   CSE-CIC-IDS2018 GENERALIZATION CLASSIFICATION REPORT{RESET}")
    print("=" * 65)
    print(report)
    print("Confusion Matrix:")
    print(f"                     Predicted BENIGN   Predicted Web Attack")
    print(f"  Actual BENIGN:         {cm[0,0]:<16}   {cm[0,1]:<16}")
    print(f"  Actual Web Attack:     {cm[1,0]:<16}   {cm[1,1]:<16}")
    print("-" * 65)
    print(f"Overall Accuracy:            {BOLD}{acc*100:.2f}%{RESET}")
    print(f"Macro F1 Score:              {BOLD}{f1*100:.2f}%{RESET}")
    print(f"Web Attack Detection Recall: {BOLD}{cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%{RESET} ({cm[1,1]}/{cm[1,0]+cm[1,1]} caught)")
    print(f"False Positive Rate:         {BOLD}{cm[0,1]/(cm[0,0]+cm[0,1])*100:.2f}%{RESET} ({cm[0,1]}/{cm[0,0]+cm[0,1]} false alarms)")
    print(f"Throughput / Latency:        {latency_ms:.3f} ms / flow ({len(X_tensor)/total_sec:,.0f} flows/sec)")
    print("=" * 65)

    # 8. Save results
    metrics_2018 = {
        "dataset": "CSE-CIC-IDS2018",
        "rows_evaluated": len(df),
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "latency_ms_per_flow": float(latency_ms)
    }
    with open(f"{args.art}/metrics_2018.json", "w") as f:
        json.dump(metrics_2018, f, indent=2)

    with open(f"{args.art}/classification_report_2018.txt", "w") as f:
        f.write(report)

    # Comparison summary with 2017
    if os.path.exists(f"{args.art}/metrics.json"):
        m2017 = json.load(open(f"{args.art}/metrics.json"))
        print(f"\n{BOLD}Comparison: 2017 (In-Domain) vs 2018 (Cross-Domain):{RESET}")
        print(f"  • CIC-IDS-2017 Test Set Macro F1:    {m2017.get('macro_f1', 0)*100:.2f}%")
        print(f"  • CSE-CIC-IDS2018 Cross-Domain F1:   {f1*100:.2f}%")
        print(f"  • Generalization Gap:                 {abs(m2017.get('macro_f1', 0) - f1)*100:.2f}%\n")


if __name__ == "__main__":
    main()
