import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42
df = pd.read_csv("/tmp/claude-0/-home-claude/40f652fc-7a57-555c-9220-46a69e64249e/scratchpad/assignment/data/student_performance/student-por.csv", sep=";")
print("Loaded shape:", df.shape)
df["target"] = (df["G3"] >= 10).astype(int)
print(df["target"].value_counts(normalize=True).round(3))

X = df.drop(columns=["G1", "G2", "G3", "target"])
y = df["target"]
cat_cols = X.select_dtypes(include="object").columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
svm_proba = svm.predict_proba(X_test_s)[:, 1]

def m(y_true, y_pred, y_proba, name):
    print(f"{name}: acc={accuracy_score(y_true,y_pred):.3f} prec={precision_score(y_true,y_pred):.3f} "
          f"rec={recall_score(y_true,y_pred):.3f} f1={f1_score(y_true,y_pred):.3f} auc={roc_auc_score(y_true,y_proba):.3f}")

m(y_test, dt_pred, dt_proba, "Decision Tree")
m(y_test, svm_pred, svm_proba, "SVM")
