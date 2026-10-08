"""
Cross-network experiments: train on CIC-IDS2017 web attacks, test on CSE-CIC-IDS2018 (22 Feb).
  A  all features, 2017 -> 2018            (baseline, reproduces the 0% result)
  B  drop shortcut features, 2017 -> 2018  (does it generalise without them?)
  C  train on 2018 split, test held-out 2018 (same-network upper bound)
  D  2017 + 2018 train split, test held-out 2018
Writes artifacts/experiments_2018/results.{json,md}. Does not touch the deployed model.
Usage: python experiments_2018.py
"""
import json, os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import DCNNBiLSTM

OUT = "artifacts/experiments_2018"; os.makedirs(OUT, exist_ok=True)
RAW17 = "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
T18 = "data/2018/test_2018_clean.csv"
SHORTCUT = ["Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward",
            "act_data_pkt_fwd", "Destination Port"]
EPOCHS = 12
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

feats = json.load(open("artifacts/meta.json"))["feature_names"]

def load17():
    df = pd.read_csv(RAW17, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True); df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    y = df["Label"].astype(str).str.contains("Web Attack").astype(int).values
    return df[feats].apply(pd.to_numeric, errors="coerce").values.astype(np.float64), y

def load18():
    df = pd.read_csv(T18, low_memory=False)
    df.columns = df.columns.str.strip()
    y = df["Label"].astype(str).str.contains("Web Attack|Brute|XSS|SQL", case=False).astype(int).values
    return df[feats].apply(pd.to_numeric, errors="coerce").values.astype(np.float64), y

def cap_benign(X, y, n=100_000, seed=0):
    rng = np.random.RandomState(seed)
    b = np.where(y == 0)[0]; a = np.where(y == 1)[0]
    b = rng.choice(b, min(n, len(b)), replace=False)
    i = np.concatenate([b, a]); return X[i], y[i]

def train_eval(Xtr, ytr, Xte, yte, cols, name):
    torch.manual_seed(42); np.random.seed(42)
    sc = StandardScaler().fit(Xtr[:, cols])
    f = lambda X: np.clip(sc.transform(X[:, cols]), -10, 10).astype(np.float32)  # clip: guard against extreme 2018 outliers
    Xt, Xe = torch.tensor(f(Xtr)), torch.tensor(f(Xte))
    yt = torch.tensor(ytr)
    cnt = torch.bincount(yt, minlength=2).float()
    w = (cnt.sum() / (2 * cnt.clamp(min=1))).clamp(max=10).to(dev)
    m = DCNNBiLSTM(len(cols), 2).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.CrossEntropyLoss(weight=w)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i+512]; opt.zero_grad()
            lf(m(Xt[b].to(dev)), yt[b].to(dev)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(m(Xe[i:i+4096].to(dev)), 1)[:, 1].cpu() for i in range(0, len(Xe), 4096)]).numpy()
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    r = dict(name=name, n_features=len(cols), test_benign=int(tn+fp), test_attacks=int(tp+fn),
             TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn),
             detection_rate=tp/max(tp+fn, 1), false_positive_rate=fp/max(fp+tn, 1),
             precision=tp/max(tp+fp, 1), roc_auc=float(roc_auc_score(yte, p)),
             pr_auc=float(average_precision_score(yte, p)))
    r["f1"] = 2*r["precision"]*r["detection_rate"]/max(r["precision"]+r["detection_rate"], 1e-9)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return r

X17, y17 = load17(); X18, y18 = load18()
print("2017:", X17.shape, y17.sum(), "attacks | 2018:", X18.shape, y18.sum(), "attacks")
X17c, y17c = cap_benign(X17, y17)
all_cols = list(range(len(feats)))
keep_cols = [i for i, n in enumerate(feats) if n not in SHORTCUT]

res = []
res.append(train_eval(X17c, y17c, X18, y18, all_cols, "A: 2017->2018, all features"))
res.append(train_eval(X17c, y17c, X18, y18, keep_cols, "B: 2017->2018, shortcut features removed"))

# C/D: hold out 30% of 2018 (stratified); caveat: random flow split, no IP grouping available
tr18, te18 = train_test_split(np.arange(len(y18)), test_size=0.3, stratify=y18, random_state=1)
res.append(train_eval(X18[tr18], y18[tr18], X18[te18], y18[te18], all_cols, "C: 2018 train -> held-out 2018"))
Xd = np.vstack([X17c, X18[tr18]]); yd = np.concatenate([y17c, y18[tr18]])
res.append(train_eval(Xd, yd, X18[te18], y18[te18], all_cols, "D: 2017+2018 train -> held-out 2018"))
# E: same as B but evaluated on same held-out 2018 slice for like-for-like with C/D
res.append(train_eval(X17c, y17c, X18[te18], y18[te18], keep_cols, "E: 2017->held-out 2018, shortcut removed"))

json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
with open(f"{OUT}/results.md", "w") as f:
    f.write("| Experiment | Attacks | Detected | Detection rate | False alarms | FPR | Precision | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in res:
        f.write(f"| {r['name']} | {r['test_attacks']} | {r['TP']} | {r['detection_rate']:.1%} | {r['FP']} | {r['false_positive_rate']:.3%} | {r['precision']:.1%} | {r['f1']:.3f} | {r['roc_auc']:.3f} |\n")
print(open(f"{OUT}/results.md").read())
