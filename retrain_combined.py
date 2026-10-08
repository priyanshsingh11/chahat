"""
Train a model on CIC-IDS2017 + 70% of CSE-CIC-IDS2018 (22 Feb); keep 30% of 2018 held out.
Writes a drop-in artifact set to artifacts_combined/ (does NOT touch artifacts/):
  sf_ids_model.pth, scaler.pkl, encoder.pkl, meta.json, held-out demo CSV + a one-flow JSON per class.
To deploy: copy those four files over artifacts/ (keep a backup of the old ones).
Usage: PYTHONPATH=. python retrain_combined.py
"""
import json, os, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from model import DCNNBiLSTM

OUT = "artifacts_combined"; os.makedirs(OUT, exist_ok=True)
RAW17 = "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
T18 = "data/2018/test_2018_clean.csv"
EPOCHS = 15
feats = json.load(open("artifacts/meta.json"))["feature_names"]
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d17 = pd.read_csv(RAW17, low_memory=False, encoding="latin-1")
d17.columns = d17.columns.str.strip()
d17 = d17.loc[:, ~d17.columns.duplicated()]
d17.replace([np.inf, -np.inf], np.nan, inplace=True); d17.dropna(inplace=True); d17.drop_duplicates(inplace=True)
d17["y"] = np.where(d17["Label"].astype(str).str.contains("Web Attack"), "Web Attack", "BENIGN")
rng = np.random.RandomState(0)
b = d17.index[d17.y == "BENIGN"]; a = d17.index[d17.y == "Web Attack"]
d17 = d17.loc[np.concatenate([rng.choice(b, min(100_000, len(b)), replace=False), a])]

d18 = pd.read_csv(T18, low_memory=False); d18.columns = d18.columns.str.strip()
d18["y"] = np.where(d18["Label"].astype(str).str.contains("Web Attack|Brute|XSS|SQL", case=False), "Web Attack", "BENIGN")
tr, te = train_test_split(np.arange(len(d18)), test_size=0.3, stratify=d18.y, random_state=1)
d18_tr, d18_te = d18.iloc[tr], d18.iloc[te]

num = lambda d: d[feats].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
Xtr = np.vstack([num(d17), num(d18_tr)]); ytr_raw = np.concatenate([d17.y.values, d18_tr.y.values])
Xte, yte_raw = num(d18_te), d18_te.y.values

enc = LabelEncoder().fit(["BENIGN", "Web Attack"])
ytr, yte = enc.transform(ytr_raw), enc.transform(yte_raw)
sc = StandardScaler().fit(Xtr)
Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32); Xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
yt = torch.tensor(ytr)

torch.manual_seed(42)
cnt = torch.bincount(yt, minlength=2).float()
w = (cnt.sum() / (2 * cnt)).clamp(max=10).to(dev)
m = DCNNBiLSTM(len(feats), 2).to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
lf = nn.CrossEntropyLoss(weight=w)
for ep in range(EPOCHS):
    m.train(); perm = torch.randperm(len(Xt)); tot = 0
    for i in range(0, len(Xt), 512):
        bi = perm[i:i+512]; opt.zero_grad()
        loss = lf(m(Xt[bi].to(dev)), yt[bi].to(dev)); loss.backward(); opt.step(); tot += loss.item() * len(bi)
    sch.step(); print(f"epoch {ep+1}/{EPOCHS} loss {tot/len(Xt):.4f}", flush=True)

m = m.cpu().eval()
with torch.no_grad():
    pred = torch.cat([m(Xe[i:i+4096]).argmax(1) for i in range(0, len(Xe), 4096)]).numpy()
rep = classification_report(yte, pred, target_names=enc.classes_, digits=4, zero_division=0)
print("\nHeld-out 2018 (never trained on):\n" + rep, confusion_matrix(yte, pred))

torch.save({"state_dict": m.state_dict(), "num_features": len(feats), "num_classes": 2}, f"{OUT}/sf_ids_model.pth")
joblib.dump(sc, f"{OUT}/scaler.pkl"); joblib.dump(enc, f"{OUT}/encoder.pkl")
json.dump({"feature_names": feats, "classes": list(enc.classes_)}, open(f"{OUT}/meta.json", "w"), indent=2)
open(f"{OUT}/heldout_2018_report.txt", "w").write(rep)

# Held-out demo: 50 attacks + 50 benign the model never saw, plus one ready-to-paste flow per class
demo = pd.concat([g.sample(min(len(g), 50), random_state=3) for _, g in d18_te.groupby("y")])
demo = demo.rename(columns={"y": "Label2"}).drop(columns=["Label"]).rename(columns={"Label2": "Label"})
demo[feats + ["Label"]].sample(frac=1, random_state=7).to_csv(f"{OUT}/demo_traffic_2018_heldout.csv", index=False)
for lab in ["Web Attack", "BENIGN"]:
    rows = d18_te[d18_te.y == lab]
    x = torch.tensor(sc.transform(num(rows)), dtype=torch.float32)
    with torch.no_grad(): p = torch.softmax(m(x), 1)[:, list(enc.classes_).index(lab)].numpy()
    r = rows.iloc[int(np.argsort(-p)[0])]  # a flow the model is confident about
    json.dump({"values": [float(r[f]) for f in feats]}, open(f"{OUT}/sample_{lab.replace(' ', '_')}.json", "w"))
print("saved to", OUT)
