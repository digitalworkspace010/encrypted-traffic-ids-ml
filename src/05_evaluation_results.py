# Evaluate the trained binary classifiers on the held-out test set.

import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_report_dir = REPORTS_DIR / "05_evaluation_results"
    evaluation_report_dir.mkdir(parents=True, exist_ok=True)

    training_summary_path = REPORTS_DIR / "04_model_training" / "model_training_summary.csv"
    test_paths = {
        ("all", True): PROCESSED_DATA_DIR / "X_test_scaled.csv",
        ("all", False): PROCESSED_DATA_DIR / "X_test.csv",
        ("top", True): PROCESSED_DATA_DIR / "X_test_selected_scaled.csv",
        ("top", False): PROCESSED_DATA_DIR / "X_test_selected.csv",
    }
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"

    required_files = [training_summary_path, y_test_path, *test_paths.values()]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required evaluation files not found. Run previous scripts first.\n"
            f"{missing_list}"
        )

    print("Loading test data and training summary...")
    y_test = pd.read_csv(y_test_path).squeeze()

    training_summary = pd.read_csv(training_summary_path)
    test_data_cache = {}

    def get_test_data(feature_set, scaled_features):
        key = (feature_set, scaled_features)
        if key not in test_data_cache:
            test_data_cache[key] = pd.read_csv(test_paths[key])
        return test_data_cache[key]

    model_files = {
        model_file.stem: model_file
        for model_file in MODELS_DIR.glob("*.joblib")
    }

    if not model_files:
        raise FileNotFoundError("No trained models found. Run 04_model_training.py first.")

    results = []

    for _, model_info in training_summary.iterrows():
        model_name = model_info["model"]
        model_file = model_files.get(model_name)

        if model_file is None:
            raise FileNotFoundError(f"Model file not found for training summary entry: {model_name}")

        feature_set = model_info["feature_set"]
        scaled_features = parse_bool(model_info["scaled_features"])
        expected_features = int(model_info["features_used"])
        imbalance_method = model_info["imbalance_method"]
        X_test = get_test_data(feature_set, scaled_features)
        test_source = test_paths[(feature_set, scaled_features)]

        if X_test.shape[1] != expected_features:
            raise ValueError(
                f"Feature count mismatch for {model_name}: "
                f"model expects {expected_features}, but {test_source} has {X_test.shape[1]}"
            )

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_file)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else None
        pr_auc = average_precision_score(y_test, y_score) if y_score is not None else None

        results.append({
            "model": model_name,
            "feature_set": feature_set,
            "scaled_features": scaled_features,
            "imbalance_method": imbalance_method,
            "test_source": str(test_source),
            "features_used": expected_features,
            "test_samples": len(y_test),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

        # Save the standard sklearn classification report.
        report = classification_report(y_test, y_pred, zero_division=0)
        with open(evaluation_report_dir / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        # Save the confusion matrix as both CSV and figure.
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Benign", "Actual Attack"],
            columns=["Predicted Benign", "Predicted Attack"]
        )
        cm_df.to_csv(METRICS_DIR / f"{model_name}_confusion_matrix.csv")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Attack"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close()

        # Some estimators expose scores, so save ROC and PR curves when possible.
        if y_score is not None:
            RocCurveDisplay.from_predictions(y_test, y_score)
            plt.title(f"ROC Curve - {model_name}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_roc_curve.png", dpi=300)
            plt.close()

            PrecisionRecallDisplay.from_predictions(y_test, y_score)
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_pr_curve.png", dpi=300)
            plt.close()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    results_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)

    with open(evaluation_report_dir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(results_df.to_string(index=False))

    print("\nEvaluation finished.")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Reports saved to: {evaluation_report_dir}")


if __name__ == "__main__":
    main()
