import json, glob, argparse, pandas as pd
from katfishnet_min.model import KatFishNetMini
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def read_jsonl(paths, label):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except:
                    continue
                txt = o.get("text") or o.get("content") or ""
                if isinstance(txt, dict):  # 방어
                    txt = txt.get("text") or txt.get("content") or ""
                if txt and isinstance(txt, str):
                    rows.append({"text": txt.strip(), "label": label})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="glob (e.g., 'data/human/*.jsonl')")
    ap.add_argument("--llm", required=True, help="glob (e.g., 'data/llm/*.jsonl')")
    ap.add_argument("--out", default="katfishnet_min.joblib")
    args = ap.parse_args()

    rows = []
    rows += read_jsonl(glob.glob(args.human), 0)
    rows += read_jsonl(glob.glob(args.llm),   1)
    df = pd.DataFrame(rows).dropna()
    print("Loaded:", df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = KatFishNetMini.new().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"AUC: {auc:.3f}")

    model.save(args.out)
    print("Saved:", args.out)