import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, recall_score, precision_score

RANDOM_STATE = 42
BASE = "/tmp/claude-0/-home-claude/40f652fc-7a57-555c-9220-46a69e64249e/scratchpad/assignment"
OUT = f"{BASE}/outputs"
import os
os.makedirs(OUT, exist_ok=True)

roc_auc_scorer = "roc_auc"

# =========================================================================
# UCI Student Performance
# =========================================================================
print("=" * 70)
print("UCI STUDENT PERFORMANCE")
print("=" * 70)
df = pd.read_csv(f"{BASE}/data/student_performance/student-por.csv", sep=";")
df["target"] = (df["G3"] >= 10).astype(int)

# keep sensitive attrs aside before encoding for fairness analysis
sensitive_uci = df[["sex", "address"]].copy()

X = df.drop(columns=["G1", "G2", "G3", "target"])
y = df["target"]
cat_cols = X.select_dtypes(include="object").columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X_enc, y, sensitive_uci, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)

# ---- 1. Single split vs 5-fold stratified CV ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE),
                          X_enc, y, cv=skf, scoring=roc_auc_scorer)
cv_f1 = cross_val_score(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE),
                         X_enc, y, cv=skf, scoring="f1")
print(f"\n5-fold stratified CV ROC-AUC: {cv_auc.round(3)} -> mean={cv_auc.mean():.3f}, std={cv_auc.std():.3f}")
print(f"5-fold stratified CV F1:      {cv_f1.round(3)} -> mean={cv_f1.mean():.3f}, std={cv_f1.std():.3f}")
print(f"(compare to single 75/25 split AUC = 0.710 as originally reported)")

# ---- 2. Learning curve ----
train_sizes, train_scores, test_scores = learning_curve(
    DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE),
    X_enc, y, cv=skf, scoring=roc_auc_scorer,
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE
)
plt.figure(figsize=(6, 4.5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1), train_scores.mean(1)+train_scores.std(1), alpha=0.15)
plt.fill_between(train_sizes, test_scores.mean(1)-test_scores.std(1), test_scores.mean(1)+test_scores.std(1), alpha=0.15)
plt.xlabel("Training set size (students)")
plt.ylabel("ROC-AUC")
plt.title("UCI Student Performance: Learning Curve (Decision Tree)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/uci_learning_curve.png", dpi=150)
plt.close()
print("\nLearning curve (train sizes):", train_sizes.astype(int))
print("Train score:", train_scores.mean(axis=1).round(3))
print("CV score:   ", test_scores.mean(axis=1).round(3))

# ---- 3. Fairness (Fairlearn) ----
y_pred_test = dt.predict(X_test)
metrics_dict = {"accuracy": accuracy_score, "selection_rate": selection_rate,
                 "recall": recall_score, "precision": precision_score}

for feat in ["sex", "address"]:
    mf = MetricFrame(metrics=metrics_dict, y_true=y_test, y_pred=y_pred_test, sensitive_features=sens_test[feat])
    print(f"\n--- UCI fairness by '{feat}' ---")
    print(mf.by_group.round(3))
    dpd = demographic_parity_difference(y_test, y_pred_test, sensitive_features=sens_test[feat])
    eod = equalized_odds_difference(y_test, y_pred_test, sensitive_features=sens_test[feat])
    print(f"Demographic parity difference: {dpd:.3f}")
    print(f"Equalized odds difference:     {eod:.3f}")
    mf.by_group.round(3).to_csv(f"{OUT}/uci_fairness_{feat}.csv")

# =========================================================================
# OULAD
# =========================================================================
print("\n" + "=" * 70)
print("OULAD")
print("=" * 70)
df_model = pd.read_csv(f"{BASE}/data/oulad_model_ready.csv")
feature_cols = ["gender", "region", "highest_education", "imd_band", "age_band",
                 "num_of_prev_attempts", "studied_credits", "disability",
                 "date_registration", "n_formative_submitted", "mean_formative_score"]
X = df_model[feature_cols]
y = df_model["target"]
sensitive_oulad = df_model[["gender", "imd_band", "disability"]].copy()

cat_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X_enc, y, sensitive_oulad, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

dt_o = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE)
dt_o.fit(X_train, y_train)

# ---- 1. CV (subsample for speed: use full X_enc, DT is fast) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc_o = cross_val_score(DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE),
                            X_enc, y, cv=skf, scoring=roc_auc_scorer)
print(f"\n5-fold stratified CV ROC-AUC: {cv_auc_o.round(3)} -> mean={cv_auc_o.mean():.3f}, std={cv_auc_o.std():.3f}")
print(f"(compare to single 75/25 split AUC = 0.947 as originally reported)")

# ---- 2. Learning curve ----
train_sizes_o, train_scores_o, test_scores_o = learning_curve(
    DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE),
    X_enc, y, cv=skf, scoring=roc_auc_scorer,
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE, n_jobs=-1
)
plt.figure(figsize=(6, 4.5))
plt.plot(train_sizes_o, train_scores_o.mean(axis=1), "o-", label="Training score")
plt.plot(train_sizes_o, test_scores_o.mean(axis=1), "o-", label="Cross-validation score")
plt.fill_between(train_sizes_o, train_scores_o.mean(1)-train_scores_o.std(1), train_scores_o.mean(1)+train_scores_o.std(1), alpha=0.15)
plt.fill_between(train_sizes_o, test_scores_o.mean(1)-test_scores_o.std(1), test_scores_o.mean(1)+test_scores_o.std(1), alpha=0.15)
plt.xlabel("Training set size (students)")
plt.ylabel("ROC-AUC")
plt.title("OULAD: Learning Curve (Decision Tree)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/oulad_learning_curve.png", dpi=150)
plt.close()
print("\nLearning curve (train sizes):", train_sizes_o.astype(int))
print("Train score:", train_scores_o.mean(axis=1).round(3))
print("CV score:   ", test_scores_o.mean(axis=1).round(3))

# ---- 3. Fairness ----
y_pred_test_o = dt_o.predict(X_test)
for feat in ["gender", "imd_band", "disability"]:
    mf = MetricFrame(metrics=metrics_dict, y_true=y_test, y_pred=y_pred_test_o, sensitive_features=sens_test[feat])
    print(f"\n--- OULAD fairness by '{feat}' ---")
    print(mf.by_group.round(3))
    dpd = demographic_parity_difference(y_test, y_pred_test_o, sensitive_features=sens_test[feat])
    eod = equalized_odds_difference(y_test, y_pred_test_o, sensitive_features=sens_test[feat])
    print(f"Demographic parity difference: {dpd:.3f}")
    print(f"Equalized odds difference:     {eod:.3f}")
    mf.by_group.round(3).to_csv(f"{OUT}/oulad_fairness_{feat}.csv")

print("\nDone. Figures + CSVs saved to", OUT)
