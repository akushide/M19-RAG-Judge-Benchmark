# -*- coding: utf-8 -*-
"""ROC-AUC analysis for the M19 Reference and Evidence judges.
Put this script beside the three input CSV files, then run:
python M19_Judge_ROC_AUC_Analysis.py
Positive class: Correct. Only Approved expert labels are included.
95% CI uses cluster bootstrap resampling by selection_id (question).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Missing package:", e)
    print("Install with: python -m pip install pandas numpy scikit-learn matplotlib openpyxl")
    sys.exit(1)

BASE = Path(__file__).resolve().parent
ref = pd.read_csv(BASE / "M19_Final_Reference_Scores.csv")
ev = pd.read_csv(BASE / "M19_Final_Evidence_Scores.csv")
key = pd.read_csv(BASE / "M19_Final_Benchmark_Private_Label_Key.csv")

for df, required, name in [
    (ref, {"case_id", "v2_reference_agreement"}, "Reference file"),
    (ev, {"case_id", "evidence_judge_raw"}, "Evidence file"),
    (key, {"response_id", "selection_id", "expected_label", "final_expert_decision"}, "Label key"),
]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")

for df, col, name in [(ref,"case_id","Reference"),(ev,"case_id","Evidence"),(key,"response_id","Label key")]:
    if df[col].duplicated().any():
        raise ValueError(f"Duplicate IDs found in {name}")

idsets = [set(ref.case_id.astype(str)), set(ev.case_id.astype(str)), set(key.response_id.astype(str))]
if not (idsets[0] == idsets[1] == idsets[2]):
    raise ValueError("The three files do not have exactly matching response IDs.")

data = key.merge(ref[["case_id","v2_reference_agreement"]], left_on="response_id", right_on="case_id", validate="one_to_one").drop(columns="case_id")
data = data.merge(ev[["case_id","evidence_judge_raw"]], left_on="response_id", right_on="case_id", validate="one_to_one").drop(columns="case_id")
data["expected_label"] = data.expected_label.astype(str).str.strip()
data["final_expert_decision"] = data.final_expert_decision.astype(str).str.strip()
data = data[(data.final_expert_decision.str.casefold()=="approved") & data.expected_label.isin(["Correct","Incorrect"])].copy()
data["y_true"] = (data.expected_label=="Correct").astype(int)
for c in ["v2_reference_agreement","evidence_judge_raw"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

def cluster_ci(df, score_col, n_boot=2000, seed=20261024):
    rng = np.random.default_rng(seed)
    groups = {g: x for g,x in df.groupby("selection_id", sort=False)}
    ids = np.array(list(groups), dtype=object)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(ids, len(ids), replace=True)
        sample = pd.concat([groups[g] for g in picked], ignore_index=True)
        if sample.y_true.nunique() < 2 or sample[score_col].nunique() < 2:
            continue
        vals.append(roc_auc_score(sample.y_true, sample[score_col]))
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(vals, [0.025,0.975])
    return float(lo), float(hi), len(vals)

specs = [
    ("Reference Judge","v2_reference_agreement"),
    ("Evidence Judge","evidence_judge_raw")
]
summary_rows, roc_rows = [], []
for name, col in specs:
    d = data.dropna(subset=["y_true","selection_id",col]).copy()
    auc = roc_auc_score(d.y_true,d[col]) if d.y_true.nunique()==2 and d[col].nunique()>1 else np.nan
    lo,hi,nboot = cluster_ci(d,col) if pd.notna(auc) else (np.nan,np.nan,0)
    # The 0.5 threshold metrics are descriptive only, not necessarily calibrated.
    pred = (d[col] >= 0.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(d.y_true,pred,labels=[0,1]).ravel()
    summary_rows.append({
        "judge":name,"score_column":col,"n_rows":len(d),
        "n_unique_questions":d.selection_id.nunique(),
        "n_correct":int(d.y_true.sum()),"n_incorrect":int((d.y_true==0).sum()),
        "roc_auc":auc,"cluster_bootstrap_95ci_low":lo,"cluster_bootstrap_95ci_high":hi,
        "valid_bootstrap_replicates":nboot,
        "accuracy_at_0.5":accuracy_score(d.y_true,pred),
        "balanced_accuracy_at_0.5":balanced_accuracy_score(d.y_true,pred),
        "precision_correct_at_0.5":precision_score(d.y_true,pred,zero_division=0),
        "recall_correct_at_0.5":recall_score(d.y_true,pred,zero_division=0),
        "f1_correct_at_0.5":f1_score(d.y_true,pred,zero_division=0),
        "TN":tn,"FP":fp,"FN":fn,"TP":tp,
        "note":"Descriptive discrimination against expert labels; not a direct measure of full factual correctness."
    })
    if pd.notna(auc):
        fpr,tpr,thresholds=roc_curve(d.y_true,d[col])
        for x,y,t in zip(fpr,tpr,thresholds):
            roc_rows.append({"judge":name,"fpr":x,"tpr":y,"threshold":t})

summary = pd.DataFrame(summary_rows)
roc_points = pd.DataFrame(roc_rows)
data["positive_class_correct"] = data.y_true
summary.to_csv(BASE/"M19_Judge_ROC_AUC_Summary.csv",index=False,encoding="utf-8-sig")
data.to_csv(BASE/"M19_Judge_ROC_AUC_Details.csv",index=False,encoding="utf-8-sig")
with pd.ExcelWriter(BASE/"M19_Judge_ROC_AUC_Analysis.xlsx",engine="openpyxl") as writer:
    summary.to_excel(writer,sheet_name="Summary",index=False)
    data.to_excel(writer,sheet_name="Merged_Details",index=False)
    roc_points.to_excel(writer,sheet_name="ROC_Curve_Points",index=False)
    key.groupby(["final_expert_decision","expected_label"]).size().reset_index(name="count").to_excel(writer,sheet_name="Label_Distribution",index=False)
if not roc_points.empty:
    plt.figure(figsize=(7,6))
    for name, g in roc_points.groupby("judge"):
        auc = summary.loc[summary.judge==name,"roc_auc"].iloc[0]
        plt.plot(g.fpr,g.tpr,label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves for M19 RAG judges"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(BASE/"M19_Judge_ROC_Curves.png",dpi=180); plt.close()

print("Rows used:",len(data),"| Unique questions:",data.selection_id.nunique())
print(summary[["judge","n_rows","n_unique_questions","roc_auc","cluster_bootstrap_95ci_low","cluster_bootstrap_95ci_high","valid_bootstrap_replicates"]].to_string(index=False))
print("\nSaved Summary CSV, Details CSV, Excel workbook, and ROC plot in:",BASE)
print("Interpret AUC cautiously: 121 responses are clustered within only 29 questions.")
