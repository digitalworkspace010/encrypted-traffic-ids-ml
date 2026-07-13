# Run the binary-classification pipeline end to end.

import subprocess
import sys
from pathlib import Path
from time import time


PIPELINE_STEPS = [
    "01_dataset_loading_exploration.py",
    "02_preprocessing.py",
    "03_feature_selection.py",
    "04_model_training.py",
    "05_evaluation_results.py",
]


def run_step(script_dir, script_name):
    script_path = script_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print("=" * 80)
    print(f"Running {script_name}")
    print("=" * 80)

    start_time = time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_dir.parent)
    elapsed = time() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode} "
            f"after {elapsed:.2f} seconds."
        )

    print(f"{script_name} completed in {elapsed:.2f} seconds.")
    return elapsed


def main():
    script_dir = Path(__file__).resolve().parent
    total_start = time()
    timings = []

    print("Starting full pipeline.")
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {script_dir.parent}")

    for script_name in PIPELINE_STEPS:
        elapsed = run_step(script_dir, script_name)
        timings.append((script_name, elapsed))

    total_elapsed = time() - total_start

    print("=" * 80)
    print("Full pipeline completed successfully.")
    print("=" * 80)

    for script_name, elapsed in timings:
        print(f"{script_name}: {elapsed:.2f} seconds")

    print(f"Total runtime: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
