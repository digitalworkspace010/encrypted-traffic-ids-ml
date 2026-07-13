# First pass over the raw CICIDS2017 files.

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASET_EXPLORATION_DIR,
    LABEL_COLUMN,
    RAW_DATA_DIR,
    SOURCE_FILE_COLUMN,
)


# Input and output folders used for the initial data audit.

DATASET_DIR = RAW_DATA_DIR
OUTPUT_DIR = DATASET_EXPLORATION_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATASET_DIR}")


# Read each raw CSV and keep the source filename for traceability.

print("Loading raw CSV files...")

dataframes = []

for file in csv_files:
    print(f"Reading {file.name}")
    df_temp = pd.read_csv(file, low_memory=False, encoding="cp1252")
    df_temp[SOURCE_FILE_COLUMN] = file.name
    dataframes.append(df_temp)

df = pd.concat(dataframes, ignore_index=True)

print("\nMerged dataset loaded.")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")


# Some CICIDS2017 columns have leading spaces, so strip names early.

df.columns = df.columns.str.strip()

print("\nColumn names stripped.")


# Quick inspection output for the dissertation audit trail.

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# Save a plain-text snapshot of the dataset structure.
with open(OUTPUT_DIR / "dataset_basic_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Total rows: {df.shape[0]}\n")
    f.write(f"Total columns: {df.shape[1]}\n\n")
    f.write(
        "Note: source_file is included only for dataset auditing "
        "and must be removed before modelling.\n\n"
    )
    f.write("Column names:\n")
    f.write("\n".join(df.columns.tolist()))
    f.write("\n\nData types:\n")
    f.write(df.dtypes.to_string())


# Work out which label column name appears in this copy of the dataset.

possible_label_cols = [LABEL_COLUMN, "label", "Class", "class", "Attack", "attack"]

label_col = None
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError("No label column found. Check dataset column names.")

print(f"\nLabel column detected: {label_col}")


# Label balance is important for the modelling stages, so save it separately.

class_counts = df[label_col].value_counts()
class_percentages = df[label_col].value_counts(normalize=True) * 100

class_distribution = pd.DataFrame({
    "count": class_counts,
    "percentage": class_percentages.round(2)
})

print("\nClass distribution:")
print(class_distribution)

class_distribution.to_csv(OUTPUT_DIR / "class_distribution.csv")


# Plot the class distribution for the report.
plt.figure(figsize=(12, 6))
class_counts.plot(kind="bar")
plt.title("CICIDS2017 Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
plt.close()


# Missing values by column.

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("\nMissing values:")
print(missing_values)

missing_values.to_csv(OUTPUT_DIR / "missing_values.csv", header=["missing_count"])


# Infinite values appear in a few flow statistics and are handled later.

numeric_df = df.select_dtypes(include=[np.number])

infinite_values = np.isinf(numeric_df).sum()
infinite_values = infinite_values[infinite_values > 0].sort_values(ascending=False)

print("\nInfinite values:")
print(infinite_values)

infinite_values.to_csv(OUTPUT_DIR / "infinite_values.csv", header=["infinite_count"])


# Check duplicates without the audit-only source_file column.

traffic_df = df.drop(columns=[SOURCE_FILE_COLUMN], errors="ignore")
duplicate_count = traffic_df.duplicated().sum()

print(f"\nDuplicate records found: {duplicate_count}")

with open(OUTPUT_DIR / "duplicate_records.txt", "w", encoding="utf-8") as f:
    f.write(f"Duplicate records found: {duplicate_count}\n")
    f.write("Duplicate check excludes source_file audit column.\n")


# Descriptive statistics, replacing infinities so describe() is meaningful.

numeric_df_clean = numeric_df.replace([np.inf, -np.inf], np.nan)
summary_stats = numeric_df_clean.describe().transpose()
summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")

print("\nSummary statistics saved using finite values only.")


# Save three versions: audit copy, small preview, and modelling input.

audit_output_path = OUTPUT_DIR / "cicids2017_merged_raw_with_source.csv"
df.to_csv(audit_output_path, index=False)

preview_output_path = OUTPUT_DIR / "cicids2017_preview_10k.csv"
df.head(10000).to_csv(preview_output_path, index=False)

merged_output_path = OUTPUT_DIR / "cicids2017_merged_raw.csv"
traffic_df.to_csv(merged_output_path, index=False)

print(f"\nAudit dataset with source_file saved to: {audit_output_path}")
print(f"Excel-friendly preview saved to: {preview_output_path}")
print(f"Modelling-safe merged raw dataset saved to: {merged_output_path}")


print("\nDataset loading and exploration finished.")
print(f"Evidence files saved in: {OUTPUT_DIR}")
