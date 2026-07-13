# Feature selection for the binary IDS experiments.

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SELECTED_FEATURES_DIR,
    RANDOM_STATE,
    CORRELATION_THRESHOLD,
    TOP_N_FEATURES,
)


def main():
    feature_report_dir = REPORTS_DIR / "03_feature_selection"
    selected_features_dir = SELECTED_FEATURES_DIR

    feature_report_dir.mkdir(parents=True, exist_ok=True)
    selected_features_dir.mkdir(parents=True, exist_ok=True)

    # Load the processed train/test files from the preprocessing stage.
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    X_train_scaled_path = PROCESSED_DATA_DIR / "X_train_scaled.csv"
    X_test_scaled_path = PROCESSED_DATA_DIR / "X_test_scaled.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"

    required_files = [
        X_train_path,
        X_test_path,
        X_train_scaled_path,
        X_test_scaled_path,
        y_train_path,
    ]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required processed files not found. Run 02_preprocessing.py first.\n"
            f"{missing_list}"
        )

    print("Loading processed training data...")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    print(f"X_train shape before feature selection: {X_train.shape}")

    # Remove one side of very highly correlated feature pairs.
    print("Checking feature correlations...")

    corr_matrix = X_train.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    correlation_pairs = (
        upper_triangle.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    high_correlation_pairs = correlation_pairs[
        correlation_pairs["correlation"] > CORRELATION_THRESHOLD
    ].sort_values(by="correlation", ascending=False)

    highly_correlated_features = sorted(high_correlation_pairs["feature_2"].unique())

    X_train_reduced = X_train.drop(columns=highly_correlated_features)

    print(f"Highly correlated features removed: {len(highly_correlated_features)}")
    print(f"X_train shape after correlation filtering: {X_train_reduced.shape}")

    pd.Series(highly_correlated_features).to_csv(
        selected_features_dir / "removed_highly_correlated_features.csv",
        index=False,
        header=["feature"],
    )

    high_correlation_pairs.to_csv(
        selected_features_dir / "high_correlation_pairs.csv",
        index=False,
    )

    # Rank the remaining features using a Random Forest importance score.
    print("Training Random Forest for feature ranking...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf.fit(X_train_reduced, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train_reduced.columns,
        "importance": rf.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        selected_features_dir / "random_forest_feature_importance.csv",
        index=False,
    )

    selected_feature_count = min(TOP_N_FEATURES, len(importance_df))
    selected_features = importance_df.head(selected_feature_count)["feature"].tolist()

    pd.Series(selected_features).to_csv(
        selected_features_dir / "selected_features.csv",
        index=False,
        header=["feature"],
    )

    print(f"Selected top {selected_feature_count} features.")

    # Save a plot of the selected features for the results chapter.
    top_features_plot = importance_df.head(selected_feature_count)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features_plot["feature"][::-1], top_features_plot["importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {selected_feature_count} Features - Random Forest Importance")
    plt.tight_layout()
    plt.savefig(feature_report_dir / "top_feature_importance.png", dpi=300)
    plt.close()

    # Save reduced train/test datasets in both scaled and unscaled form.
    X_test = pd.read_csv(X_test_path)
    X_train_scaled = pd.read_csv(X_train_scaled_path)
    X_test_scaled = pd.read_csv(X_test_scaled_path)

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    X_train_selected_scaled = X_train_scaled[selected_features]
    X_test_selected_scaled = X_test_scaled[selected_features]

    X_train_selected.to_csv(PROCESSED_DATA_DIR / "X_train_selected.csv", index=False)
    X_test_selected.to_csv(PROCESSED_DATA_DIR / "X_test_selected.csv", index=False)
    X_train_selected_scaled.to_csv(
        PROCESSED_DATA_DIR / "X_train_selected_scaled.csv",
        index=False,
    )
    X_test_selected_scaled.to_csv(
        PROCESSED_DATA_DIR / "X_test_selected_scaled.csv",
        index=False,
    )

    # Save the feature-selection audit report.
    with open(feature_report_dir / "feature_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("Feature Selection Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Initial number of features: {X_train.shape[1]}\n")
        f.write(f"Correlation threshold: {CORRELATION_THRESHOLD}\n")
        f.write(f"Highly correlated feature pairs found: {len(high_correlation_pairs)}\n")
        f.write(f"Highly correlated features removed: {len(highly_correlated_features)}\n")
        f.write(f"Features after correlation filtering: {X_train_reduced.shape[1]}\n")
        f.write(f"Requested top features: {TOP_N_FEATURES}\n")
        f.write(f"Top features selected using Random Forest: {selected_feature_count}\n\n")

        f.write("Selected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")

        f.write("\nOutput files:\n")
        f.write("- removed_highly_correlated_features.csv\n")
        f.write("- high_correlation_pairs.csv\n")
        f.write("- random_forest_feature_importance.csv\n")
        f.write("- selected_features.csv\n")
        f.write("- X_train_selected.csv\n")
        f.write("- X_test_selected.csv\n")
        f.write("- X_train_selected_scaled.csv\n")
        f.write("- X_test_selected_scaled.csv\n")
        f.write("- top_feature_importance.png\n")

    print("Feature selection finished.")
    print(f"Reports saved to: {feature_report_dir}")
    print(f"Selected features saved to: {selected_features_dir}")


if __name__ == "__main__":
    main()
