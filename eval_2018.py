"""
SF-IDS - Cross-dataset test: evaluate the 2017-trained model on CSE-CIC-IDS2018.

The model, scaler and encoder are used EXACTLY as trained on CIC-IDS-2017
(no retraining, no refitting). This measures how well the detector generalises
to a different network recorded a year later.

Usage:
    python eval_2018.py data/2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
    python eval_2018.py "data/2018/*.csv"
    python eval_2018.py "data/2018/*.csv" --max-benign 300000   # sample benign for speed
    python eval_2018.py "data/2018/*.csv" --fill-missing         # zero-fill any unmapped column

Outputs go to artifacts/eval_2018/ (report, metrics.json, confusion matrix, comparison).
"""
import argparse
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from model import DCNNBiLSTM

ART = "artifacts"
OUT = f"{ART}/eval_2018"

# CIC-IDS-2017 feature name  ->  CSE-CIC-IDS2018 feature name
COL_MAP = {
    "Destination Port": "Dst Port",
    "Flow Duration": "Flow Duration",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "CWE Flag Count": "CWE Flag Count",
    "ECE Flag Count": "ECE Flag Cnt",
    "Down/Up Ratio": "Down/Up Ratio",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
}

# CSE-CIC-IDS2018 label -> CIC-IDS-2017 label (as cleaned by preprocess.py)
LABEL_MAP = {
    "benign": "BENIGN",
    "brute force -web": "Web Attack - Brute Force",
    "brute force-web": "Web Attack - Brute Force",
    "brute force -xss": "Web Attack - XSS",
    "brute force-xss": "Web Attack - XSS",
    "sql injection": "Web Attack - Sql Injection",
}


def norm(name: str) -> str:
    """Loose key for column matching: lowercase, letters/digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_columns(feature_names, cols_2018, fill_missing):
    """For each 2017 feature the model expects, find the matching 2018 column."""
    lookup = {norm(c): c for c in cols_2018}
    chosen, missing = {}, []
    for f in feature_names:
        base = re.sub(r"\.\d+$", "", f)          # 'Fwd Header Length.1' duplicate -> base name
        candidates = [COL_MAP.get(base), base]
        hit = next((lookup[norm(c)] for c in candidates if c and norm(c) in lookup), None)
        if hit is None:
            missing.append(f)
        chosen[f] = hit
    if missing:
        msg = f"{len(missing)} model features not found in the 2018 file: {missing}"
        if not fill_missing:
            raise SystemExit(msg + "\nRe-run with --fill-missing to zero-fill them (and mention it in the report).")
        print("WARNING:", msg, "-> filled with 0")
    return chosen, missing


def load_2018(paths, max_benign, seed=42):
    frames = []
    for p in paths:
        print(f"Loading {p} ...")
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[df["Label"].astype(str).str.strip() != "Label"]   # repeated header rows in 2018 CSVs
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    raw = df["Label"].astype(str).str.strip()
    df["Label17"] = raw.str.lower().map(LABEL_MAP)
    unknown = raw[df["Label17"].isna()].value_counts()
    if len(unknown):
        print("Skipping labels the model was never trained on:\n" + unknown.to_string())
    df = df[df["Label17"].notna()]

    benign = df[df["Label17"] == "BENIGN"]
    if max_benign and len(benign) > max_benign:
        df = pd.concat([benign.sample(max_benign, random_state=seed),
                        df[df["Label17"] != "BENIGN"]])
        print(f"Sampled {max_benign:,} of {len(benign):,} benign flows (all attacks kept)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-benign", type=int, default=0, help="0 = use all benign rows")
    ap.add_argument("--fill-missing", action="store_true")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    paths = sorted({p for a in args.paths for p in glob.glob(a)})
    if not paths:
        raise SystemExit("No files matched. Example: python eval_2018.py \"data/2018/*.csv\"")
    os.makedirs(OUT, exist_ok=True)

    # --- the 2017 model, exactly as trained -----------------------------------
    meta = json.load(open(f"{ART}/meta.json"))
    features, classes = meta["feature_names"], meta["classes"]
    scaler = joblib.load(f"{ART}/scaler.pkl")
    ckpt = torch.load(f"{ART}/sf_ids_model.pth", map_location="cpu")
    model = DCNNBiLSTM(ckpt["num_features"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    merged = "Web Attack" in classes
    print(f"Model classes: {classes} ({'merged web attacks' if merged else 'separate web attacks'})")

    # --- 2018 data, renamed into 2017 feature order ---------------------------
    df = load_2018(paths, args.max_benign)
    chosen, missing = resolve_columns(features, df.columns, args.fill_missing)
    X = pd.DataFrame({f: (pd.to_numeric(df[c], errors="coerce") if c else 0.0)
                      for f, c in chosen.items()})
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values
    X = X[ok]
    y_lbl = df["Label17"].astype(str)[ok]
    if merged:
        y_lbl = y_lbl.where(~y_lbl.str.startswith("Web Attack"), "Web Attack")
    y_lbl = y_lbl.tolist()
    print(f"Evaluating {len(X):,} flows ({(~ok).sum():,} dropped for inf/NaN)")
    print(pd.Series(y_lbl).value_counts().to_string())

    cls_index = {c: i for i, c in enumerate(classes)}
    y_true = np.array([cls_index[l] for l in y_lbl])

    # --- predict in batches -----------------------------------------------------
    Xs = scaler.transform(X.values)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), args.batch):
            xb = torch.tensor(Xs[i:i + args.batch], dtype=torch.float32)
            preds.append(model(xb).argmax(1).numpy())
    y_pred = np.concatenate(preds)

    # --- metrics ------------------------------------------------------------------
    labels = list(range(len(classes)))
    present = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   zero_division=0, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=present,
                                                  average="macro", zero_division=0)
    benign_i = cls_index["BENIGN"]
    is_att_true, is_att_pred = y_true != benign_i, y_pred != benign_i
    tp = int((is_att_true & is_att_pred).sum())
    fp = int((~is_att_true & is_att_pred).sum())
    fn = int((is_att_true & ~is_att_pred).sum())
    tn = int((~is_att_true & ~is_att_pred).sum())
    binary = {
        "attack_recall (detection rate)": tp / max(tp + fn, 1),
        "attack_precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "attacks_detected": tp, "attacks_missed": fn,
        "false_alarms": fp, "benign_correct": tn,
    }
    metrics = {"dataset": "CSE-CIC-IDS2018", "files": paths, "flows": int(len(X)),
               "accuracy": accuracy_score(y_true, y_pred), "macro_precision": p,
               "macro_recall": r, "macro_f1": f1, "attack_vs_benign": binary,
               "zero_filled_features": missing, "confusion_matrix": cm.tolist(),
               "classes": classes}
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    open(f"{OUT}/classification_report.txt", "w").write(report)

    print("\n=== CSE-CIC-IDS2018 (model trained on CIC-IDS-2017, no retraining) ===")
    print(report)
    for k, v in binary.items():
        print(f"  {k:32} {v:.4f}" if isinstance(v, float) else f"  {k:32} {v:,}")

    # --- comparison with the 2017 test-set results --------------------------------
    rows = [("Macro F1", "macro_f1"), ("Accuracy", "accuracy"),
            ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall")]
    lines = ["| Metric | 2017 test set | 2018 (unseen network) |", "|---|---|---|"]
    try:
        m17 = json.load(open(f"{ART}/metrics.json"))
        for name, key in rows:
            lines.append(f"| {name} | {m17[key]:.4f} | {metrics[key]:.4f} |")
    except FileNotFoundError:
        for name, key in rows:
            lines.append(f"| {name} | n/a | {metrics[key]:.4f} |")
    lines += [f"| Attack detection rate | - | {binary['attack_recall (detection rate)']:.4f} |",
              f"| Attack precision | - | {binary['attack_precision']:.4f} |",
              f"| False-positive rate | - | {binary['false_positive_rate']:.4f} |"]
    table = "\n".join(lines)
    open(f"{OUT}/comparison.md", "w").write(table + "\n")
    print("\n" + table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(classes)
        normed = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 2.5))
        ax.imshow(normed, cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks(range(n), classes, rotation=45, ha="right")
        ax.set_yticks(range(n), classes)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if normed[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Trained on CIC-IDS-2017, tested on CSE-CIC-IDS2018")
        fig.tight_layout()
        fig.savefig(f"{OUT}/confusion_matrix_2018.png", dpi=150)
    except Exception as e:
        print("Skipped plot:", e)

    print(f"\nSaved results to {OUT}/")


if __name__ == "__main__":
    main()
