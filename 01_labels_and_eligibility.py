"""Label derivation + eligibility filter (verbatim logic from the training notebook, cell 9).
Train: image_labels_train.csv has 3 rows/image (one per radiologist); a finding is positive if mean vote >= 2/3.
Test:  image_labels_test.csv has 1 consensus row/image (0/1); threshold 0.5 on that single row.
"""
import pandas as pd, numpy as np, argparse
CLASS_NAMES = ["Tuberculosis", "Pneumonia", "Normal"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
TRAIN_VOTE_THRESHOLD = 2 / 3
SEED = 42
MAX_TRAIN_NORMAL = 500

def aggregate_image_labels(df, is_train):
    work = df[["image_id", "Tuberculosis", "Pneumonia", "No finding"]].copy()
    for c in ["Tuberculosis", "Pneumonia", "No finding"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    agg = work.groupby("image_id", as_index=False)[["Tuberculosis", "Pneumonia", "No finding"]].mean()
    thr = TRAIN_VOTE_THRESHOLD if is_train else 0.5
    for c in ["Tuberculosis", "Pneumonia", "No finding"]:
        agg[f"{c}_positive"] = agg[c] >= thr
    def assign(r):
        tb, pn, nf = bool(r["Tuberculosis_positive"]), bool(r["Pneumonia_positive"]), bool(r["No finding_positive"])
        if tb and pn: return "DROP_TB_PNEUMONIA_COPOSITIVE"
        if tb and not pn and not nf: return "Tuberculosis"
        if pn and not tb and not nf: return "Pneumonia"
        if nf and not tb and not pn: return "Normal"
        return "DROP_OTHER_OR_INCONSISTENT"
    agg["label"] = agg.apply(assign, axis=1)
    return agg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True); ap.add_argument("--test_csv", required=True)
    a = ap.parse_args()
    tr = aggregate_image_labels(pd.read_csv(a.train_csv), True)
    te = aggregate_image_labels(pd.read_csv(a.test_csv), False)
    tr = tr[tr.label.isin(CLASS_NAMES)]; te = te[te.label.isin(CLASS_NAMES)].copy()
    normal = tr[tr.label == "Normal"]; disease = tr[tr.label != "Normal"]
    if len(normal) > MAX_TRAIN_NORMAL: normal = normal.sample(n=MAX_TRAIN_NORMAL, random_state=SEED)
    tr = pd.concat([disease, normal], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    tr["target"] = tr.label.map(CLASS_TO_IDX); te["target"] = te.label.map(CLASS_TO_IDX)
    tr.to_csv("train_manifest.csv", index=False); te.to_csv("test_manifest.csv", index=False)
    print(tr.label.value_counts(), te.label.value_counts())  # expect 416/405/500 and 128/210/2051
