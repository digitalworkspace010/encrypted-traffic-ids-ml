# Train the binary classifiers used in the IDS experiments.

import pandas as pd
import joblib
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
)


def main():
    report_dir = REPORTS_DIR / "04_model_training"
    report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load both all-feature and selected-feature training sets.

    X_train_all_tree_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_train_all_scaled_path = PROCESSED_DATA_DIR / "X_train_scaled.csv"
    X_train_tree_path = PROCESSED_DATA_DIR / "X_train_selected.csv"
    X_train_scaled_path = PROCESSED_DATA_DIR / "X_train_selected_scaled.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"

    required_files = [
        X_train_all_tree_path,
        X_train_all_scaled_path,
        X_train_tree_path,
        X_train_scaled_path,
        y_train_path,
    ]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required training files not found. Run 03_feature_selection.py first.\n"
            f"{missing_list}"
        )

    print("Loading training data...")

    X_train_all_tree = pd.read_csv(X_train_all_tree_path)
    X_train_all_scaled = pd.read_csv(X_train_all_scaled_path)
    X_train_tree = pd.read_csv(X_train_tree_path)
    X_train_scaled = pd.read_csv(X_train_scaled_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    print(f"X_train all features shape: {X_train_all_tree.shape}")
    print(f"X_train all features scaled shape: {X_train_all_scaled.shape}")
    print(f"X_train selected shape: {X_train_tree.shape}")
    print(f"X_train selected scaled shape: {X_train_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")

    class_counts = y_train.value_counts().sort_index()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    scale_pos_weight = (
        negative_count / positive_count
        if positive_count
        else 1.0
    )

    # Define the experiment grid. Linear models use scaled features; tree models do not.

    models = {
        "logistic_regression_all_features_baseline": {
            "features": X_train_all_scaled,
            "feature_source": str(X_train_all_scaled_path),
            "feature_set": "all",
            "scaled": True,
            "imbalance_method": "none",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        },
        "logistic_regression_baseline": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "none",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        },
        "logistic_regression_smote": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "SMOTE",
            "estimator": Pipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
        },
        "logistic_regression_class_weight": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "class_weight=balanced",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        },

        "svm_linear_all_features_baseline": {
            "features": X_train_all_scaled,
            "feature_source": str(X_train_all_scaled_path),
            "feature_set": "all",
            "scaled": True,
            "imbalance_method": "none",
            "estimator": LinearSVC(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        },
        "svm_linear_baseline": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "none",
            "estimator": LinearSVC(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        },
        "svm_linear_smote": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "SMOTE",
            "estimator": Pipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", LinearSVC(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                )),
            ]),
        },
        "svm_linear_class_weight": {
            "features": X_train_scaled,
            "feature_source": str(X_train_scaled_path),
            "feature_set": "top",
            "scaled": True,
            "imbalance_method": "class_weight=balanced",
            "estimator": LinearSVC(
                max_iter=5000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        },

        "random_forest_all_features_baseline": {
            "features": X_train_all_tree,
            "feature_source": str(X_train_all_tree_path),
            "feature_set": "all",
            "scaled": False,
            "imbalance_method": "none",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "random_forest_baseline": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "none",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "random_forest_smote": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "SMOTE",
            "estimator": Pipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
        },
        "random_forest_class_weight": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "class_weight=balanced",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
        },

        "xgboost_all_features_baseline": {
            "features": X_train_all_tree,
            "feature_source": str(X_train_all_tree_path),
            "feature_set": "all",
            "scaled": False,
            "imbalance_method": "none",
            "estimator": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        },
        "xgboost_baseline": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "none",
            "estimator": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        },
        "xgboost_smote": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "SMOTE",
            "estimator": Pipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                )),
            ]),
        },
        "xgboost_scale_pos_weight": {
            "features": X_train_tree,
            "feature_source": str(X_train_tree_path),
            "feature_set": "top",
            "scaled": False,
            "imbalance_method": "scale_pos_weight",
            "estimator": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
        },
    }

    # Train each model and keep a compact summary for evaluation.

    training_summary = []

    for model_name, model_config in models.items():
        print(f"\nTraining {model_name}...")

        X_train = model_config["features"]
        estimator = model_config["estimator"]
        start_time = time()

        estimator.fit(X_train, y_train)

        training_time_seconds = time() - start_time

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(estimator, model_path)

        training_summary.append({
            "model": model_name,
            "saved_path": str(model_path),
            "feature_source": model_config["feature_source"],
            "feature_set": model_config["feature_set"],
            "scaled_features": model_config["scaled"],
            "imbalance_method": model_config["imbalance_method"],
            "features_used": X_train.shape[1],
            "training_samples": X_train.shape[0],
            "training_time_seconds": round(training_time_seconds, 2),
        })

        print(f"{model_name} saved to: {model_path}")
        print(f"Training time: {training_time_seconds:.2f} seconds")

    # Save the training summary and a readable report.

    summary_df = pd.DataFrame(training_summary)
    summary_df.to_csv(report_dir / "model_training_summary.csv", index=False)

    with open(report_dir / "model_training_report.txt", "w", encoding="utf-8") as f:
        f.write("Model Training Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"All features used: {X_train_all_tree.shape[1]}\n")
        f.write(f"Top features used: {X_train_tree.shape[1]}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Class 0 training samples: {negative_count}\n")
        f.write(f"Class 1 training samples: {positive_count}\n")
        f.write(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}\n\n")

        f.write("Feature inputs:\n")
        f.write(f"- All-feature tree-based baseline models: {X_train_all_tree_path}\n")
        f.write(f"- All-feature scale-sensitive baseline models: {X_train_all_scaled_path}\n")
        f.write(f"- Top-feature tree-based models: {X_train_tree_path}\n")
        f.write(f"- Top-feature scale-sensitive models: {X_train_scaled_path}\n\n")

        f.write("Experiment scenarios:\n")
        f.write("- All features + no imbalance handling\n")
        f.write("- Top features + no imbalance handling\n")
        f.write("- Top features + SMOTE\n")
        f.write("- Top features + class weighting / scale_pos_weight\n\n")

        f.write("Imbalance handling scenarios:\n")
        f.write("- Baseline: no imbalance handling\n")
        f.write("- SMOTE: applied inside imblearn pipeline on training data only\n")
        f.write("- Class weighting: estimator-level class weighting where supported\n")
        f.write("- XGBoost weighting: scale_pos_weight\n\n")

        f.write("Models trained:\n")
        for item in training_summary:
            f.write(
                f"- {item['model']} | "
                f"features={item['feature_set']} | "
                f"imbalance={item['imbalance_method']} | "
                f"scaled={item['scaled_features']} | "
                f"time={item['training_time_seconds']}s | "
                f"{item['saved_path']}\n"
            )

    print("\nModel training finished.")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Training reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
