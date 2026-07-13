# Optional multiclass comparison using the selected binary features.

import json
from time import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import (
    LABEL_COLUMN,
    MERGED_RAW_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    RESULTS_DIR,
    SELECTED_FEATURES_DIR,
    TEST_SIZE,
)


REPORT_DIR = REPORTS_DIR / "07_multiclass_comparison"
METRICS_DIR = RESULTS_DIR / "multiclass_metrics"
FIGURES_DIR = RESULTS_DIR / "multiclass_figures"
MODELS_DIR = RESULTS_DIR / "multiclass_models"


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def clean_dataset(df):
    df.columns = df.columns.str.strip()

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found.")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    # Keep the cleaning rules aligned with the binary preprocessing stage.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def save_confusion_matrix(model_name, y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in class_names],
        columns=[f"Predicted {label}" for label in class_names],
    )
    cm_df.to_csv(METRICS_DIR / f"{model_name}_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix - {model_name}")
    plt.colorbar(label="Recall within actual class")
    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=90)
    plt.yticks(tick_positions, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_normalized_confusion_matrix.png", dpi=300)
    plt.close()


def save_per_class_recall_plot(per_class_recall_df):
    pivot = per_class_recall_df.pivot(
        index="class",
        columns="model",
        values="recall",
    )

    pivot.plot(kind="bar", figsize=(14, 7))
    plt.ylabel("Recall")
    plt.title("Multiclass Per-Class Recall Comparison")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multiclass_per_class_recall_comparison.png", dpi=300)
    plt.close()


def save_metric_comparison_plot(results_df):
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    plot_df = results_df.set_index("model")[metric_cols]

    plot_df.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Score")
    plt.title("Multiclass Model Comparison")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multiclass_model_metric_comparison.png", dpi=300)
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    selected_features_path = SELECTED_FEATURES_DIR / "selected_features.csv"

    required_files = [MERGED_RAW_DATA_PATH, selected_features_path]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required multiclass input files not found. Run scripts 01-03 first.\n"
            f"{missing_list}"
        )

    print("Loading merged dataset for multiclass comparison...")
    df = pd.read_csv(MERGED_RAW_DATA_PATH, low_memory=False)
    initial_shape = df.shape
    df = clean_dataset(df)
    cleaned_shape = df.shape

    selected_features = pd.read_csv(selected_features_path)["feature"].tolist()
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Selected features not found in multiclass dataframe: {missing_features}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COLUMN])
    class_names = label_encoder.classes_.tolist()

    X = df[selected_features]

    class_distribution = pd.DataFrame({
        "class": class_names,
        "encoded_label": range(len(class_names)),
        "count": pd.Series(y).value_counts().sort_index().values,
    })
    class_distribution["percentage"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    ).round(4)
    class_distribution.to_csv(METRICS_DIR / "multiclass_label_distribution.csv", index=False)

    mapping_df = pd.DataFrame({
        "encoded_label": range(len(class_names)),
        "class": class_names,
    })
    mapping_df.to_csv(METRICS_DIR / "multiclass_label_mapping.csv", index=False)
    joblib.dump(label_encoder, MODELS_DIR / "multiclass_label_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "random_forest_multiclass_baseline": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest_multiclass_class_weight": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "xgboost_multiclass_baseline": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    per_class_recall_rows = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time()
        model.fit(X_train, y_train)
        training_time = time() - start_time

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        report_text = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

        with open(REPORT_DIR / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as file:
            file.write(report_text)

        write_json(REPORT_DIR / f"{model_name}_classification_report.json", report_dict)

        for class_name in class_names:
            per_class_recall_rows.append({
                "model": model_name,
                "class": class_name,
                "recall": report_dict[class_name]["recall"],
                "precision": report_dict[class_name]["precision"],
                "f1_score": report_dict[class_name]["f1-score"],
                "support": report_dict[class_name]["support"],
            })

        save_confusion_matrix(model_name, y_test, y_pred, class_names)

        results.append({
            "model": model_name,
            "feature_set": "binary_selected_top_30",
            "features_used": X_train.shape[1],
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "classes": len(class_names),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "training_time_seconds": round(training_time, 2),
            "model_path": str(model_path),
        })

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    per_class_recall_df = pd.DataFrame(per_class_recall_rows)

    results_df.to_csv(METRICS_DIR / "multiclass_model_comparison_metrics.csv", index=False)
    per_class_recall_df.to_csv(METRICS_DIR / "multiclass_per_class_metrics.csv", index=False)

    save_metric_comparison_plot(results_df)
    save_per_class_recall_plot(per_class_recall_df)

    best_macro = results_df.iloc[0]
    best_weighted = results_df.sort_values("weighted_f1", ascending=False).iloc[0]

    with open(REPORT_DIR / "multiclass_comparison_report.md", "w", encoding="utf-8") as file:
        file.write("# Multiclass Model Comparison Report\n\n")
        file.write("This report compares the two strongest binary model families on multiclass attack-type classification.\n\n")
        file.write("Models evaluated:\n\n")
        file.write("- Random Forest baseline\n")
        file.write("- Random Forest with class weighting\n")
        file.write("- XGBoost baseline\n\n")
        file.write("The experiment uses the 30 selected features from the binary feature-selection stage.\n\n")
        file.write("## Dataset Summary\n\n")
        file.write(f"- Initial merged shape: {initial_shape}\n")
        file.write(f"- Cleaned shape: {cleaned_shape}\n")
        file.write(f"- Training samples: {X_train.shape[0]}\n")
        file.write(f"- Test samples: {X_test.shape[0]}\n")
        file.write(f"- Features used: {X_train.shape[1]}\n")
        file.write(f"- Number of classes: {len(class_names)}\n\n")
        file.write("## Best Models\n\n")
        file.write(f"- Best by macro F1: `{best_macro['model']}` ({best_macro['macro_f1'] * 100:.2f}%).\n")
        file.write(f"- Best by weighted F1: `{best_weighted['model']}` ({best_weighted['weighted_f1'] * 100:.2f}%).\n\n")
        file.write("## Model Comparison\n\n")
        file.write("```text\n")
        file.write(results_df.to_string(index=False))
        file.write("\n```")
        file.write("\n\n")
        file.write("## Important Figures\n\n")
        file.write("![Metric Comparison](../../results/multiclass_figures/multiclass_model_metric_comparison.png)\n\n")
        file.write("![Per-Class Recall Comparison](../../results/multiclass_figures/multiclass_per_class_recall_comparison.png)\n\n")
        file.write("## Output Files\n\n")
        file.write("- `results/multiclass_metrics/multiclass_model_comparison_metrics.csv`\n")
        file.write("- `results/multiclass_metrics/multiclass_per_class_metrics.csv`\n")
        file.write("- `results/multiclass_metrics/multiclass_label_distribution.csv`\n")
        file.write("- `results/multiclass_metrics/multiclass_label_mapping.csv`\n")
        file.write("- `results/multiclass_models/*.joblib`\n")
        file.write("- `results/multiclass_figures/*.png`\n")

    print("\nMulticlass comparison finished.")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
