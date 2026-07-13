# Preprocess the merged CICIDS2017 dataset for binary classification.

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    MERGED_RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    LABEL_COLUMN,
    SOURCE_FILE_COLUMN,
)


def main():
    input_file = MERGED_RAW_DATA_PATH
    output_report_dir = REPORTS_DIR / "02_preprocessing"

    output_report_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Merged dataset not found: {input_file}")

    print("Loading merged raw dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    df.columns = df.columns.str.strip()

    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found.")

    # Drop the audit column before modelling, if the exploration script left it in.
    if SOURCE_FILE_COLUMN in df.columns:
        df = df.drop(columns=[SOURCE_FILE_COLUMN])

    # Keep the before-cleaning counts for the report.
    numeric_df = df.select_dtypes(include=[np.number])

    inf_count = np.isinf(numeric_df).sum().sum()
    nan_count_before = df.isna().sum().sum()
    duplicate_count_before = df.duplicated().sum()

    # Treat infinities like missing values; they cannot be scaled or modelled safely.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Remove exact duplicate traffic rows.
    df = df.drop_duplicates()

    cleaned_shape = df.shape
    print(f"Shape after cleaning: {cleaned_shape}")

    # Binary target: BENIGN = 0, any attack label = 1.
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    df["binary_label"] = df[LABEL_COLUMN].apply(
        lambda label: 0 if label.upper() == "BENIGN" else 1
    )

    class_distribution = df["binary_label"].value_counts().sort_index()
    class_distribution_percent = (
        df["binary_label"].value_counts(normalize=True).sort_index() * 100
    )

    print("\nBinary class distribution after cleaning:")
    print(pd.DataFrame({
        "count": class_distribution,
        "percentage": class_distribution_percent.round(2)
    }))

    # Split features and target.
    X = df.drop(columns=[LABEL_COLUMN, "binary_label"])
    y = df["binary_label"]

    # Keep numeric feature columns only.
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X = X.select_dtypes(include=[np.number])

    # Stratify so the minority class ratio is preserved in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train_distribution = y_train.value_counts().sort_index()
    y_train_distribution_percent = (
        y_train.value_counts(normalize=True).sort_index() * 100
    )
    y_test_distribution = y_test.value_counts().sort_index()
    y_test_distribution_percent = (
        y_test.value_counts(normalize=True).sort_index() * 100
    )

    # Fit the scaler after splitting to avoid leaking test-set statistics.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    # Save both raw numeric features and scaled versions.
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    X_train_scaled.to_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Save the fitted scaler and feature list for later stages.
    joblib.dump(scaler, PROCESSED_DATA_DIR / "standard_scaler.joblib")

    pd.Series(X.columns).to_csv(
        PROCESSED_DATA_DIR / "feature_names.csv",
        index=False,
        header=["feature"],
    )

    # Write a short audit report for the methodology chapter.
    with open(output_report_dir / "preprocessing_report.txt", "w", encoding="utf-8") as f:
        f.write("Preprocessing Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Input file: {input_file}\n")
        f.write(f"Initial shape: {initial_shape}\n")
        f.write(f"Final cleaned shape: {cleaned_shape}\n\n")

        f.write("Cleaning summary:\n")
        f.write(f"- Infinite values detected before cleaning: {inf_count}\n")
        f.write(f"- NaN values before cleaning: {nan_count_before}\n")
        f.write(f"- Duplicate rows detected before cleaning: {duplicate_count_before}\n")
        f.write(f"- Rows removed during cleaning: {initial_shape[0] - cleaned_shape[0]}\n\n")

        f.write("Binary label encoding:\n")
        f.write("- BENIGN = 0\n")
        f.write("- All attack classes = 1\n\n")

        f.write("Class distribution after cleaning:\n")
        for label, count in class_distribution.items():
            percentage = class_distribution_percent[label]
            f.write(f"- Class {label}: {count} records ({percentage:.2f}%)\n")

        f.write("\nFeature selection for modelling:\n")
        f.write(f"- Numeric features retained: {len(X.columns)}\n")
        f.write(f"- Non-numeric feature columns removed: {len(non_numeric_columns)}\n")
        if non_numeric_columns:
            for column in non_numeric_columns:
                f.write(f"  - {column}\n")

        f.write("\nTrain/test split:\n")
        f.write(f"- Test size: {TEST_SIZE}\n")
        f.write(f"- Random state: {RANDOM_STATE}\n")
        f.write("- Stratified split: Yes\n\n")

        f.write("Training set class distribution:\n")
        for label, count in y_train_distribution.items():
            percentage = y_train_distribution_percent[label]
            f.write(f"- Class {label}: {count} records ({percentage:.2f}%)\n")

        f.write("\nTest set class distribution:\n")
        for label, count in y_test_distribution.items():
            percentage = y_test_distribution_percent[label]
            f.write(f"- Class {label}: {count} records ({percentage:.2f}%)\n")
        f.write("\n")

        f.write("Prevention of data leakage:\n")
        f.write("- NaN and infinite values removed before splitting\n")
        f.write("- Duplicate records removed before splitting\n")
        f.write("- StandardScaler fitted only on training data\n")
        f.write("- Test data transformed using training scaler parameters\n\n")

        f.write("Output files:\n")
        f.write("- X_train.csv\n")
        f.write("- X_test.csv\n")
        f.write("- X_train_scaled.csv\n")
        f.write("- X_test_scaled.csv\n")
        f.write("- y_train.csv\n")
        f.write("- y_test.csv\n")
        f.write("- standard_scaler.joblib\n")
        f.write("- feature_names.csv\n")

    print("\nPreprocessing finished.")
    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")
    print(f"Report saved to: {output_report_dir}")


if __name__ == "__main__":
    main()
