import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from xgboost import XGBClassifier

from config import (
    FILE_HOLDOUT_FILENAME,
    LABEL_COLUMN,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SOURCE_FILE_COLUMN,
)


DATA_DIR = RAW_DATA_DIR
OUTPUT_DIR = REPORTS_DIR / "08_file_holdout_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_FILE = FILE_HOLDOUT_FILENAME

frames = []

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df[SOURCE_FILE_COLUMN] = file.name
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Clean invalid values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Remove duplicate records without using source_file as part of the duplicate check
duplicate_subset = [col for col in df.columns if col != SOURCE_FILE_COLUMN]
df.drop_duplicates(subset=duplicate_subset, inplace=True)

# Binary label: BENIGN = 0, all attacks = 1
df["binary_label"] = df[LABEL_COLUMN].apply(
    lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
)

# Use numerical features only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in ["binary_label"]]

X = df[feature_cols]
y = df["binary_label"]

train_mask = df[SOURCE_FILE_COLUMN] != HOLDOUT_FILE
test_mask = df[SOURCE_FILE_COLUMN] == HOLDOUT_FILE

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Holdout file": HOLDOUT_FILE,
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-score (%)": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "PR-AUC (%)": round(average_precision_score(y_test, y_score) * 100, 2),
        "Train samples": len(X_train),
        "Test samples": len(X_test)
    })

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv(
    OUTPUT_DIR / "file_holdout_robustness_results.csv",
    index=False
)
