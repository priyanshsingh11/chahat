"""
SF-IDS MVP - Step 3: Fog-node inference API (FastAPI)
Run:  fastapi dev main.py      (or: uvicorn main:app --port 8000)
Docs: http://127.0.0.1:8000/docs
"""
import json
import os
import time
from collections import Counter, deque

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import DCNNBiLSTM

ART = "artifacts"
CONFIDENCE_TO_BLOCK = 0.60

meta = json.load(open(f"{ART}/meta.json"))
FEATURES = meta["feature_names"]
scaler = joblib.load(f"{ART}/scaler.pkl")
encoder = joblib.load(f"{ART}/encoder.pkl")
ckpt = torch.load(f"{ART}/sf_ids_model.pth", map_location="cpu")
model = DCNNBiLSTM(ckpt["num_features"], ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

app = FastAPI(title="SF-IDS Fog Node", version="0.1-mvp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recent_alerts: deque = deque(maxlen=100)
stats = Counter()


@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SF-IDS Fog Node</h1><p>Visit <a href='/docs'>/docs</a></p>"


class Flow(BaseModel):
    # Either {"features": {"Flow Duration": 123, ...}} or {"values": [ ... in order ... ]}
    features: dict[str, float] | None = None
    values: list[float] | None = None
    src: str | None = None  # optional source id/IP for the alert log


def to_vector(flow: Flow) -> np.ndarray:
    if flow.values is not None:
        if len(flow.values) != len(FEATURES):
            raise HTTPException(400, f"Expected {len(FEATURES)} values, got {len(flow.values)}")
        return np.asarray(flow.values, dtype=np.float64).reshape(1, -1)
    if flow.features is not None:
        missing = [f for f in FEATURES if f not in flow.features]
        if missing:
            raise HTTPException(400, f"Missing {len(missing)} features, e.g. {missing[:3]}")
        return np.asarray([[flow.features[f] for f in FEATURES]], dtype=np.float64)
    raise HTTPException(400, "Send 'features' (dict) or 'values' (list)")


@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES), "classes": list(encoder.classes_)}


@app.post("/predict")
def predict(flow: Flow):
    t = time.perf_counter()
    x = torch.tensor(scaler.transform(to_vector(flow)), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    label = str(encoder.inverse_transform([idx])[0])
    conf = float(probs[idx])
    malicious = label.upper() != "BENIGN"
    action = "BLOCK" if malicious and conf >= CONFIDENCE_TO_BLOCK else (
        "REVIEW" if malicious else "ALLOW")
    latency = (time.perf_counter() - t) * 1000

    stats[action] += 1
    result = {"classification": label, "confidence": round(conf, 4),
              "action": action, "latency_ms": round(latency, 2)}
    if malicious:
        recent_alerts.appendleft({**result, "src": flow.src, "ts": time.time()})
    return result


@app.get("/alerts")
def alerts():
    return {"counts": dict(stats), "recent": list(recent_alerts)}


@app.post("/explain")
def explain(alert: dict):
    """Optional GenAI layer: turn a flagged alert into a short analyst note.
    Uses Claude if ANTHROPIC_API_KEY is set, otherwise a built-in template."""
    label = alert.get("classification", "unknown")
    conf = alert.get("confidence", 0)
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model=os.getenv("EXPLAIN_MODEL", "claude-haiku-5-5"),
                max_tokens=300,
                messages=[{"role": "user", "content":
                           f"An intrusion detection model flagged a network flow as '{label}' "
                           f"with confidence {conf}. In under 120 words for a SOC analyst: what "
                           f"this attack is, why it matters, and 2-3 recommended actions."}])
            return {"source": "llm", "explanation": msg.content[0].text}
        except Exception as e:
            return {"source": "template", "error": str(e), "explanation": template(label)}
    return {"source": "template", "explanation": template(label)}


def template(label: str) -> str:
    notes = {
        "DDoS": "Distributed denial of service: many sources flooding the service. Rate-limit at the gateway and block offending IPs.",
        "DoS": "Denial of service flood against a single service. Throttle the source and check autoscaling cost limits.",
        "PortScan": "Reconnaissance: a host probing many ports. Block the source and review exposed services.",
        "Brute Force": "Repeated login attempts. Lock the account, enforce MFA, and block the source.",
        "XSS": "Cross-site scripting payload. Sanitise inputs and enable WAF XSS rules.",
        "Sql Injection": "SQL injection attempt. Use parameterised queries and enable WAF SQLi rules.",
        "Bot": "Botnet traffic. Isolate the host and check for malware.",
        "Infiltration": "Possible internal compromise. Isolate the host and start incident response.",
    }
    for k, v in notes.items():
        if k.lower() in label.lower():
            return f"{label}: {v}"
    return f"{label}: anomalous traffic pattern. Review the source and consider blocking it."
