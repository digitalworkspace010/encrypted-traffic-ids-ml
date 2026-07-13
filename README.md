# Machine Learning Intrusion Detection for Encrypted Traffic

This repository contains the code for my MSc dissertation experiments on intrusion detection in encrypted network traffic. Because encrypted traffic does not allow payload inspection, the project uses flow-based CICIDS2017 features and tests whether machine-learning models can still identify malicious activity from metadata and statistical patterns.

The pipeline covers:

- loading and exploring the CICIDS2017 CSV files;
- cleaning the data and creating a binary benign/attack label;
- selecting a smaller feature set;
- training and evaluating binary classifiers;
- running an additional multiclass attack-type comparison.

## Dataset

The CICIDS2017 CSV files are not included in this repository. Download them separately and place them in:

```text
data/raw/
```

The `data/raw/`, `data/processed/`, `reports/`, and `results/` folders are kept in the repository with `.gitkeep` files so the folder structure is visible on GitHub. The `.gitkeep` files are only placeholders.

Expected raw files include:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## Setup

The code was written for Python 3.10 or newer.

Install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The main packages used are:

```text
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
joblib
```

Note that `imbalanced-learn` is installed under that package name, but imported in Python as `imblearn`.

## Project Structure

```text
.
|-- data/
|   |-- raw/          put CICIDS2017 CSV files here
|   `-- processed/    created by preprocessing
|-- reports/          created during pipeline runs
|-- results/          created during pipeline runs
`-- src/
    |-- config.py
    |-- 01_dataset_loading_exploration.py
    |-- 02_preprocessing.py
    |-- 03_feature_selection.py
    |-- 04_model_training.py
    |-- 05_evaluation_results.py
    |-- 06_run_full_pipeline.py
    `-- 07_multiclass_comparison.py
```

## Running the Binary Pipeline

After placing the CICIDS2017 CSV files in `data/raw/`, run:

```powershell
python .\src\06_run_full_pipeline.py
```

This runs the first five scripts in order:

```text
01_dataset_loading_exploration.py
02_preprocessing.py
03_feature_selection.py
04_model_training.py
05_evaluation_results.py
```

The full run can take some time because the training stage compares several algorithms and imbalance-handling approaches.

## Running the Multiclass Comparison

The multiclass script should be run after scripts `01` to `03`, because it reuses the merged dataset and selected features:

```powershell
python .\src\07_multiclass_comparison.py
```

This trains multiclass versions of:

- Random Forest;
- Random Forest with class weighting;
- XGBoost.

## Script Notes

### 01 Dataset Loading and Exploration

Input:

```text
data/raw/*.csv
```

This script merges the raw CICIDS2017 CSV files, strips column-name whitespace, adds a temporary `source_file` column for auditing, checks labels and data quality, and saves a modelling-safe merged CSV without `source_file`.

Main output:

```text
reports/01_dataset_exploration/
```

### 02 Preprocessing

This stage removes infinite values, missing values, and duplicate rows. It then converts the original labels into a binary target:

```text
BENIGN -> 0
attack -> 1
```

It also creates stratified train/test splits and saves both scaled and unscaled feature files.

Main outputs:

```text
data/processed/
reports/02_preprocessing/preprocessing_report.txt
```

### 03 Feature Selection

This script removes highly correlated features, ranks the remaining features with Random Forest feature importance, and keeps the top 30 features. The selected feature list is applied to both scaled and unscaled train/test sets.

Main outputs:

```text
results/selected_features/
reports/03_feature_selection/
```

### 04 Model Training

This stage trains Logistic Regression, Linear SVM, Random Forest, and XGBoost models. It compares all-feature baselines, selected-feature baselines, SMOTE, and class weighting or `scale_pos_weight`.

Main outputs:

```text
results/models/
reports/04_model_training/
```

### 05 Evaluation Results

This script loads the trained binary models, selects the correct test feature file for each one, and calculates accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. It also saves confusion matrices, ROC curves, precision-recall curves, and classification reports.

Main outputs:

```text
results/metrics/
results/figures/
reports/05_evaluation_results/
```

### 07 Multiclass Comparison

This optional stage keeps the original CICIDS2017 attack labels, encodes them as multiclass targets, reuses the selected 30 features, and compares Random Forest and XGBoost models on attack-type classification.

Main outputs:

```text
results/multiclass_metrics/
results/multiclass_figures/
results/multiclass_models/
reports/07_multiclass_comparison/
```

## Experiment Design

The binary experiments compare four main settings:

```text
1. All features + no imbalance handling
2. Top features + no imbalance handling
3. Top features + SMOTE
4. Top features + class weighting / scale_pos_weight
```

The multiclass extension focuses on Random Forest and XGBoost because these were the strongest model families in the binary experiments.

## Metrics

The dataset is imbalanced, so accuracy is reported but not treated as the only measure of performance.

Binary metrics:

- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC.

Multiclass metrics:

- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- per-class recall.

Metric values are stored as proportions. For example:

```text
0.9990 = 99.90%
```

## Reproducibility

The main reproducibility settings are defined in `src/config.py`:

```text
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

Small differences can still occur across machines because Random Forest and XGBoost use parallel training.

## Output Files

After running the pipeline, the main generated folders are:

```text
reports/01_dataset_exploration/
reports/02_preprocessing/
reports/03_feature_selection/
reports/04_model_training/
reports/05_evaluation_results/
reports/07_multiclass_comparison/
results/selected_features/
results/metrics/
results/multiclass_metrics/
results/figures/
results/multiclass_figures/
results/models/
results/multiclass_models/
```

These outputs include selected feature tables, metrics, confusion matrices, figures, trained models, binary experiment results, and multiclass experiment results.

## Method Notes

- `source_file` is used only for checking where rows came from and is removed before modelling.
- Feature selection is performed using the training data only.
- Scaling is fitted on the training data only.
- SMOTE is applied inside the training pipeline only.
- The test set is held out until final evaluation.
