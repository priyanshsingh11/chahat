"""
SF-IDS MVP - Step 4: Live attack simulator (the demo)
Replays held-out real traffic (benign + attacks) against the running API and
prints each verdict, then a scoreboard.

Usage:  python attack_simulator.py                  (default http://127.0.0.1:8000)
        python attack_simulator.py --url https://<your-endpoint> --n 200 --delay 0.05
"""
import argparse
import random
import time
import pandas as pd
import requests

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv("artifacts/demo_traffic.csv").head(args.n)
    features = [c for c in df.columns if c != "Label"]
    correct, blocked, latencies = 0, 0, []

    print(f"Replaying {len(df)} flows against {args.url}/predict\n")
    for i, row in df.iterrows():
        payload = {"values": row[features].astype(float).tolist(),
                   "src": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"}
        r = requests.post(f"{args.url}/predict", json=payload, timeout=10).json()
        truth, pred, action = row["Label"], r["classification"], r["action"]
        ok = truth == pred
        correct += ok
        blocked += action == "BLOCK"
        latencies.append(r["latency_ms"])
        color = GREEN if action == "ALLOW" else RED if action == "BLOCK" else YELLOW
        mark = "✓" if ok else "✗"
        print(f"{mark} {color}{action:6}{RESET} pred={pred:<28} truth={truth:<28} "
              f"conf={r['confidence']:.2f} {r['latency_ms']:.1f}ms")
        time.sleep(args.delay)

    lat = sorted(latencies)
    print(f"\nAccuracy on replayed flows: {correct}/{len(df)} = {correct/len(df):.1%}")
    print(f"Blocked: {blocked} | median latency {lat[len(lat)//2]:.1f} ms | "
          f"p95 {lat[int(len(lat)*0.95)-1]:.1f} ms")


if __name__ == "__main__":
    main()
