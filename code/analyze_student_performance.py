"""
Case Study: Chisholm Institute Data Analyst role
Dataset 2/2: UCI Student Performance (Portuguese language course, student-por.csv)
Source: https://archive.ics.uci.edu/dataset/320/student+performance (CC BY 4.0)

Task framing: EARLY prediction of at-risk students (pass/fail final outcome),
using only demographic / family / lifestyle / school-engagement features that
would be available BEFORE final grades are known (G1, G2, G3 excluded from
features; G3 used only to derive the target). This mirrors a real retention/
resourcing use case: a Data Analyst wants to flag at-risk students early
enough to act, not just report outcomes after the fact.

Algorithms: Decision Tree Classifier + Support Vector Machine (SVM)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
OUT_FIG = "/home/claude/case_study/outputs/figures"
OUT_RES = "/home/claude/case_study/outputs/results"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/case_study/data/student_performance/student-por.csv", sep=";")
print("Loaded shape:", df.shape)

# ---------------------------------------------------------------
# 2. Target: Pass (G3 >= 10) vs Fail (G3 < 10)
# ---------------------------------------------------------------
df["target"] = (df["G3"] >= 10).astype(int)
print("Class balance (1=Pass, 0=Fail):")
print(df["target"].value_counts(normalize=True).round(3))

# Drop grade leakage columns and raw target
X = df.drop(columns=["G1", "G2", "G3", "target"])
y = df["target"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"\n{len(cat_cols)} categorical cols, {len(num_cols)} numeric cols")

X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 3. Decision Tree
# ---------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 4. SVM (needs scaling)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
svm_proba = svm.predict_proba(X_test_s)[:, 1]

# ---------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------
def get_metrics(y_true, y_pred, y_proba, name):
    m = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return m

results = [
    get_metrics(y_test, dt_pred, dt_proba, "Decision Tree"),
    get_metrics(y_test, svm_pred, svm_proba, "SVM (RBF)"),
]
results_df = pd.DataFrame(results).round(3)
print("\n=== UCI Student Performance: Model comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_RES}/uci_student_performance_metrics.csv", index=False)

with open(f"{OUT_RES}/uci_classification_reports.txt", "w") as f:
    f.write("DECISION TREE\n" + classification_report(y_test, dt_pred) + "\n\n")
    f.write("SVM (RBF)\n" + classification_report(y_test, svm_pred))

# ---------------------------------------------------------------
# 6. Feature importance (Decision Tree) - top 10
# ---------------------------------------------------------------
importances = pd.Series(dt.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
top10 = importances.head(10)
print("\nTop 10 features (Decision Tree importance):")
print(top10)
top10.to_csv(f"{OUT_RES}/uci_top_features.csv")

plt.figure(figsize=(7, 5))
top10.iloc[::-1].plot(kind="barh", color="#4C72B0")
plt.title("UCI Student Performance: Top 10 Feature Importances (Decision Tree)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/uci_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, name in zip(axes, [dt_pred, svm_pred], ["Decision Tree", "SVM (RBF)"]):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Fail", "Pass"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/uci_confusion_matrices.png", dpi=150)
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
plt.title("UCI Student Performance: ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/uci_roc_curves.png", dpi=150)
plt.close()

print("\nSaved figures to", OUT_FIG)
print("Saved results to", OUT_RES)
