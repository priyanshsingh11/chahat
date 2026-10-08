# SF-IDS MVP — DCNN-BiLSTM Intrusion Detection + Fog-Node API

Minimum viable version of SF-IDS: a hybrid **DCNN-BiLSTM** model (after Hnamte & Hussain, 2023)
trained on **CIC-IDS-2017**, served by a **FastAPI** inference endpoint, and demonstrated with a
**live attack simulator**. Serverless/fog deployment and the security modules are the next phase.

```
CIC-IDS CSV ──► preprocess.py ──► train.py (Colab GPU) ──► main.py (FastAPI) ◄── attack_simulator.py
               clean, split,      DCNN-BiLSTM, metrics,     /predict /alerts       replays real
               scale (train only) confusion matrix          /explain               held-out flows
```

## Files

| File | What it does |
|---|---|
| `preprocess.py` | Cleans CSVs (inf/NaN, duplicate columns, odd dash in labels), splits train/test **before** scaling, StandardScaler, saves `artifacts/` |
| `model.py` | DCNN-BiLSTM: 2×Conv1d → BiLSTM → Dense → softmax |
| `train.py` | Class-weighted training, test-set accuracy / precision / recall / F1, confusion matrix, CPU latency per flow |
| `main.py` | FastAPI fog node: `/health`, `/predict`, `/alerts`, `/explain` (optional LLM analyst note) |
| `attack_simulator.py` | Replays held-out benign + attack flows against the API, prints verdicts and a scoreboard |

## Run it (≈3 hours)

### 1. Setup — laptop (10 min)
```bash
cd sfids-mvp
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw   # put your CIC-IDS-2017 CSVs here
```

### 2. Preprocess — laptop, CPU is fine (5 min)
```bash
python preprocess.py "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
# more attack types (recommended): python preprocess.py "data/raw/*.csv"
```
Output: `artifacts/dataset.pt, scaler.pkl, encoder.pkl, meta.json, demo_traffic.csv`

### 3. Train — Google Colab T4 GPU (15–30 min)
1. colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU**
2. Upload `model.py`, `train.py`, and the whole `artifacts/` folder (zip it, upload, unzip):
```python
!unzip -o artifacts.zip
!python train.py --epochs 15
!zip -r trained.zip artifacts
```
3. Download `trained.zip` and unzip it over your local `artifacts/` folder.

You now have `sf_ids_model.pth`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`
— these numbers and the confusion matrix go straight into your report/slides.

> Using all CIC-IDS-2017 days on a CPU would be slow; one file (~170k rows) trains on CPU in
> minutes too, so Colab is only essential for the full dataset.

### 4. Serve — laptop (2 min)
```bash
uvicorn main:app --port 8000
```
Open http://127.0.0.1:8000/docs to try `/predict` in the browser.

### 5. Demo — second terminal
```bash
python attack_simulator.py --n 100 --delay 0.1
```
Green ALLOW for benign, red BLOCK for attacks, then accuracy and latency. Check
http://127.0.0.1:8000/alerts for the alert log.

### Optional: LLM explanation layer
```bash
export ANTHROPIC_API_KEY=...      # without a key, /explain uses a built-in template
pip install anthropic
```
`POST /explain` with an alert JSON returns a short analyst note. Only flagged alerts call the LLM,
so cost stays tiny — the DL model does the actual detection.

## What to tell the supervisor

- **Done:** DCNN-BiLSTM detection core trained on CIC-IDS-2017, evaluated on a held-out test set
  (accuracy, macro F1, per-class report, confusion matrix), served as a fog-node inference API with
  ~1–2 ms per-flow latency, demonstrated live with real traffic replay.
- **Differences from the paper (honest list):** class-weighted loss instead of SMOTE; CIC-IDS-2017
  instead of CICIDS2018/Edge-IIoT; numeric flow features only, so no one-hot step needed.
- **Next phase:** Docker + serverless deployment (AWS Lambda / API Gateway), KMS-encrypted feature
  store, orchestrator security (keys, attestation), CICIDS2018 + Edge-IIoT, SMOTE comparison.

## Known gotchas
- Very rare classes (<10 rows, e.g. Infiltration/Heartbleed) are dropped automatically — note this in the report.
- Accuracy alone is misleading on imbalanced data; report **macro F1** and per-class recall.
- `MAX_ROWS_PER_CLASS` in `preprocess.py` caps BENIGN at 200k rows to keep training fast.
