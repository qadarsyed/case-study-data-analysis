import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42
DATA = "/tmp/claude-0/-home-claude/40f652fc-7a57-555c-9220-46a69e64249e/scratchpad/assignment/data/oulad"

info = pd.read_csv(f"{DATA}/studentInfo.csv")
reg = pd.read_csv(f"{DATA}/studentRegistration.csv")
stu_assess = pd.read_csv(f"{DATA}/studentAssessment.csv")
assess_meta = pd.read_csv(f"{DATA}/assessments.csv")

print("studentInfo:", info.shape)
key = ["code_module", "code_presentation", "id_student"]
reg_small = reg[key + ["date_registration"]].copy()
reg_small["date_registration"] = pd.to_numeric(reg_small["date_registration"], errors="coerce")

assess_meta_f = assess_meta[assess_meta["assessment_type"].isin(["TMA", "CMA"])]
sa = stu_assess.merge(assess_meta_f[["id_assessment", "code_module", "code_presentation", "assessment_type"]],
                       on="id_assessment", how="inner")
sa["score"] = pd.to_numeric(sa["score"], errors="coerce")

agg = sa.groupby(["code_module", "code_presentation", "id_student"]).agg(
    n_formative_submitted=("score", "count"),
    mean_formative_score=("score", "mean"),
).reset_index()

df = info.merge(reg_small, on=key, how="left").merge(agg, on=key, how="left")
df["n_formative_submitted"] = df["n_formative_submitted"].fillna(0)
df["mean_formative_score"] = df["mean_formative_score"].fillna(0)

df["target"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
print("Class balance:\n", df["target"].value_counts(normalize=True).round(3))

feature_cols = ["gender", "region", "highest_education", "imd_band", "age_band",
                 "num_of_prev_attempts", "studied_credits", "disability",
                 "date_registration", "n_formative_submitted", "mean_formative_score"]
df_model = df[feature_cols + ["target"]].dropna()
print(f"Rows after dropping missing: {len(df_model)} / {len(df)}")

X = df_model[feature_cols]
y = df_model["target"]
cat_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

SVM_TRAIN_N = 8000
X_train_svm, _, y_train_svm, _ = train_test_split(X_train, y_train, train_size=SVM_TRAIN_N, random_state=RANDOM_STATE, stratify=y_train)
scaler = StandardScaler()
X_train_svm_s = scaler.fit_transform(X_train_svm)
X_test_s = scaler.transform(X_test)
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
svm.fit(X_train_svm_s, y_train_svm)
svm_pred = svm.predict(X_test_s)
svm_proba = svm.predict_proba(X_test_s)[:, 1]

def m(y_true, y_pred, y_proba, name):
    print(f"{name}: acc={accuracy_score(y_true,y_pred):.3f} prec={precision_score(y_true,y_pred):.3f} "
          f"rec={recall_score(y_true,y_pred):.3f} f1={f1_score(y_true,y_pred):.3f} auc={roc_auc_score(y_true,y_proba):.3f}")

m(y_test, dt_pred, dt_proba, "Decision Tree")
m(y_test, svm_pred, svm_proba, "SVM (subsample 8000)")

importances = pd.Series(dt.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
print("\nTop 5 features:\n", importances.head(5))

# save processed data + splits for reuse (learning curve, fairness)
df_model.to_csv("/tmp/claude-0/-home-claude/40f652fc-7a57-555c-9220-46a69e64249e/scratchpad/assignment/data/oulad_model_ready.csv", index=False)
