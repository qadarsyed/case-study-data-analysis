"""
Case Study: Chisholm Institute Data Analyst role
Dataset 1/2: OULAD (Open University Learning Analytics Dataset)
Source: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset (CC BY 4.0)

Task framing: EARLY identification of at-risk students (Withdrawn/Fail vs
Pass/Distinction), using demographics, registration timing, and FORMATIVE
assessment behaviour only (TMA/CMA submissions) -- final Exam scores are
excluded from features because they occur at course end and would leak
the outcome. Scope note: the raw VLE clickstream log (studentVle.csv,
~433MB uncompressed) was excluded due to file-size/time constraints for
this assignment; this is documented as a limitation.

Algorithms: Decision Tree Classifier + Support Vector Machine (SVM)
(same two algorithms as the UCI Student Performance analysis, enabling
direct comparison of insights across the two data sources)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
DATA = "/home/claude/case_study/data/oulad"
OUT_FIG = "/home/claude/case_study/outputs/figures"
OUT_RES = "/home/claude/case_study/outputs/results"

# ---------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------
info = pd.read_csv(f"{DATA}/studentInfo.csv")
reg = pd.read_csv(f"{DATA}/studentRegistration.csv")
stu_assess = pd.read_csv(f"{DATA}/studentAssessment.csv")
assess_meta = pd.read_csv(f"{DATA}/assessments.csv")

print("studentInfo:", info.shape)
print("final_result distribution:")
print(info["final_result"].value_counts())

key = ["code_module", "code_presentation", "id_student"]

# Registration: keep date_registration only (date_unregistration leaks outcome)
reg_small = reg[key + ["date_registration"]].copy()
reg_small["date_registration"] = pd.to_numeric(reg_small["date_registration"], errors="coerce")

# Assessment behaviour: only formative assessments (TMA/CMA), exclude Exam (end-of-course, leaks outcome)
assess_meta_f = assess_meta[assess_meta["assessment_type"].isin(["TMA", "CMA"])]
sa = stu_assess.merge(assess_meta_f[["id_assessment", "code_module", "code_presentation", "assessment_type"]],
                       on="id_assessment", how="inner")
sa["score"] = pd.to_numeric(sa["score"], errors="coerce")

agg = sa.groupby(["code_module", "code_presentation", "id_student"]).agg(
    n_formative_submitted=("score", "count"),
    mean_formative_score=("score", "mean"),
).reset_index()

df = info.merge(reg_small, on=key, how="left").merge(agg, on=key, how="left")

# Students with no formative submissions -> 0 submitted, score set to 0 (did not engage)
df["n_formative_submitted"] = df["n_formative_submitted"].fillna(0)
df["mean_formative_score"] = df["mean_formative_score"].fillna(0)

# ---------------------------------------------------------------
# 2. Target: AtRisk = Withdrawn or Fail (1), Pass or Distinction (0)
# ---------------------------------------------------------------
df["target"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
print("\nClass balance (1=At risk, 0=Not at risk):")
print(df["target"].value_counts(normalize=True).round(3))

feature_cols = ["gender", "region", "highest_education", "imd_band", "age_band",
                 "num_of_prev_attempts", "studied_credits", "disability",
                 "date_registration", "n_formative_submitted", "mean_formative_score"]
df_model = df[feature_cols + ["target"]].dropna()
print(f"\nRows after dropping missing: {len(df_model)} / {len(df)}")

X = df_model[feature_cols]
y = df_model["target"]

cat_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 3. Decision Tree (full training set)
# ---------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 4. SVM -- subsample for tractability (documented decision)
# ---------------------------------------------------------------
SVM_TRAIN_N = 8000
if len(X_train) > SVM_TRAIN_N:
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train, y_train, train_size=SVM_TRAIN_N, random_state=RANDOM_STATE, stratify=y_train
    )
else:
    X_train_svm, y_train_svm = X_train, y_train

scaler = StandardScaler()
X_train_svm_s = scaler.fit_transform(X_train_svm)
X_test_s = scaler.transform(X_test)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
svm.fit(X_train_svm_s, y_train_svm)
svm_pred = svm.predict(X_test_s)
svm_proba = svm.predict_proba(X_test_s)[:, 1]

# ---------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------
def get_metrics(y_true, y_pred, y_proba, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

results = [
    get_metrics(y_test, dt_pred, dt_proba, "Decision Tree"),
    get_metrics(y_test, svm_pred, svm_proba, "SVM (RBF, subsample n=8000)"),
]
results_df = pd.DataFrame(results).round(3)
print("\n=== OULAD: Model comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_RES}/oulad_metrics.csv", index=False)

with open(f"{OUT_RES}/oulad_classification_reports.txt", "w") as f:
    f.write("DECISION TREE\n" + classification_report(y_test, dt_pred) + "\n\n")
    f.write("SVM (RBF)\n" + classification_report(y_test, svm_pred))

# ---------------------------------------------------------------
# 6. Feature importance (Decision Tree) - top 10
# ---------------------------------------------------------------
importances = pd.Series(dt.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
top10 = importances.head(10)
print("\nTop 10 features (Decision Tree importance):")
print(top10)
top10.to_csv(f"{OUT_RES}/oulad_top_features.csv")

plt.figure(figsize=(7, 5))
top10.iloc[::-1].plot(kind="barh", color="#DD8452")
plt.title("OULAD: Top 10 Feature Importances (Decision Tree)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/oulad_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, name in zip(axes, [dt_pred, svm_pred], ["Decision Tree", "SVM (RBF)"]):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not at risk", "At risk"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not at risk", "At risk"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/oulad_confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. ROC curves
# ---------------------------------------------------------------
plt.figure(figsize=(5.5, 5))
for proba, name in [(dt_proba, "Decision Tree"), (svm_proba, "SVM (RBF)")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("OULAD: ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/oulad_roc_curves.png", dpi=150)
plt.close()

print("\nSaved figures to", OUT_FIG)
print("Saved results to", OUT_RES)
