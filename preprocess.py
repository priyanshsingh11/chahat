"""
SF-IDS MVP - Step 1: Preprocessing
Reads one or more CIC-IDS CSV files, cleans them, splits train/test BEFORE
scaling (no data leakage), standard-scales features (as in Hnamte & Hussain 2023),
and saves everything the trainer and the API need.

Usage:
    python preprocess.py data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    python preprocess.py data/raw/*.csv          # use several days at once
"""
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUT_DIR = "artifacts"
MIN_CLASS_SAMPLES = 10      # drop classes too rare to split/learn
MAX_ROWS_PER_CLASS = 200_000  # cap huge BENIGN class so training stays fast


def load(paths):
    frames = []
    for p in paths:
        print(f"Loading {p} ...")
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    merge_web = "--merge-web" in sys.argv
    paths = []
    for arg in sys.argv[1:]:
        if arg == "--merge-web":
            continue
        paths.extend(glob.glob(arg))
    if not paths:
        sys.exit("Give at least one CSV path, e.g. python preprocess.py data/raw/*.csv [--merge-web]")

    df = load(paths)

    # --- Clean -------------------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()]           # e.g. 'Fwd Header Length.1'
    df["Label"] = df["Label"].astype(str).str.strip()
    # Real CIC-IDS-2017 web-attack labels contain an odd dash byte: 'Web Attack \x96 XSS'
    df["Label"] = (df["Label"].str.replace(r"[\x96–�]", "-", regex=True)
                   .str.replace(r"\s+", " ", regex=True))
    if merge_web:
        df["Label"] = df["Label"].apply(lambda l: "Web Attack" if "Web Attack" in str(l) else l)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    y_raw = df["Label"]
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().all()]                      # keep numeric columns only
    X = X.loc[:, X.std() > 0]                          # drop constant columns
    feature_names = list(X.columns)

    # Drop ultra-rare classes, cap the majority class
    counts = y_raw.value_counts()
    keep = counts[counts >= MIN_CLASS_SAMPLES].index
    mask = y_raw.isin(keep)
    X, y_raw = X[mask], y_raw[mask]
    idx = pd.Index([])
    for _, g in y_raw.groupby(y_raw):
        idx = idx.append(g.sample(min(len(g), MAX_ROWS_PER_CLASS), random_state=42).index)
    X, y_raw = X.loc[idx], y_raw.loc[idx]

    print("\nClass distribution:")
    print(y_raw.value_counts().to_string())

    # --- Encode + split (stratified) -----------------------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42)

    # --- Scale: fit on TRAIN only ------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Save ----------------------------------------------------------------
    torch.save({
        "X_train": torch.tensor(X_train_s, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_test": torch.tensor(X_test_s, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
    }, f"{OUT_DIR}/dataset.pt")
    joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
    joblib.dump(encoder, f"{OUT_DIR}/encoder.pkl")
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump({"feature_names": feature_names,
                   "classes": list(encoder.classes_)}, f, indent=2)

    # Raw (unscaled) held-out rows for the live attack simulator
    demo = pd.DataFrame(X_test, columns=feature_names)
    demo["Label"] = encoder.inverse_transform(y_test)
    demo = pd.concat([g.sample(min(len(g), 50), random_state=1)
                      for _, g in demo.groupby("Label")])
    demo.sample(frac=1, random_state=7).to_csv(f"{OUT_DIR}/demo_traffic.csv", index=False)

    print(f"\nFeatures: {len(feature_names)} | Classes: {list(encoder.classes_)}")
    print(f"Train: {X_train_s.shape} | Test: {X_test_s.shape}")
    print(f"Saved to ./{OUT_DIR}/ (dataset.pt, scaler.pkl, encoder.pkl, meta.json, demo_traffic.csv)")


if __name__ == "__main__":
    main()
