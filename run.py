import argparse
import os
import sys
import subprocess
import yaml
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional

# --- Bolt's Refactor: Direct Imports ---
# We are replacing the slow and brittle `subprocess` calls with direct
# function imports. This makes the pipeline a cohesive Python application,
# not just a series of disconnected scripts. It's faster, more robust,
# and allows for better state management and error handling.
from scripts.validate_raw import run_validation
from scripts.clean_dataset import run_cleaning
from scripts.prepare_data import run_preparation
from training.train import run_training

def check_dependencies() -> None:
    """
    Checks if all required packages from requirements.txt are installed.
    Exits the script if any dependencies are missing.
    """
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('-e')]
    except FileNotFoundError:
        print("Warning: requirements.txt not found. Skipping dependency check.", file=sys.stderr)
        return

    missing_packages = []
    for package in requirements:
        package_name = package.split('==')[0].strip()
        if not package_name:
            continue
        try:
            version(package_name)
        except PackageNotFoundError:
            missing_packages.append(package_name)

    if missing_packages:
        print("The following required packages are not installed:", file=sys.stderr)
        for package in missing_packages:
            print(f" - {package}", file=sys.stderr)
        print("\nPlease install them by running: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    """
    Main entrypoint for the miniature-memory pipeline.

    This script serves as a unified orchestrator for the entire data processing
    and model training pipeline. It has been refactored by Bolt to use direct
    function calls instead of `subprocess`, making it faster and more robust.

    The pipeline consists of the following stages:
    1.  Validation: Checks raw data for quality and language content.
    2.  Cleaning: Filters ads, normalizes text, and removes invalid characters.
    3.  Preparation: Concatenates cleaned files into a single training corpus.
    4.  Training: Trains the GPT model on the prepared corpus.

    Each stage can be controlled via command-line arguments, and the script
    also handles synchronization with the Hugging Face Hub.
    """
    parser = argparse.ArgumentParser(
        description="A unified orchestrator for the miniature-memory pipeline."
    )
    # --- Data Pipeline Arguments ---
    parser.add_argument("--raw-dir", type=str, default="dataset/raw", help="Directory for raw data.")
    parser.add_argument("--cleaned-dir", type=str, default="dataset/cleaned", help="Directory for cleaned data.")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Directory for processed data.")

    # --- Pipeline Control Arguments ---
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip the raw data validation step."
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip the dataset cleaning step."
    )
    parser.add_argument(
        "--skip-preparation", action="store_true", help="Skip the data preparation step."
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip the model training step."
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Disable Hugging Face Hub synchronization."
    )

    # --- Training Specific Argument ---
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/small.yaml",
        help="Path to the training configuration file."
    )
    args = parser.parse_args()

    # --- Hub Synchronization (Pre-run) ---
    if not args.no_sync:
        print("--- Synchronizing with Hugging Face Hub (Pulling) ---")
        try:
            subprocess.run(["python3", "scripts/sync_hub.py", "pull", "--target", "all"], check=True)
            print("--- Synchronization (Pull) complete ---\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not pull from Hugging Face Hub. Continuing with local state. Error: {e}\n", file=sys.stderr)
    else:
        print("--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---\n")

    check_dependencies()

    print("Starting the miniature-memory pipeline...\n")

    # --- Step 1: Validation ---
    if not args.skip_validation:
        print("--- Running Validation ---")
        run_validation(args.raw_dir)
        print("--- Validation completed successfully ---\n")

    # --- Step 2: Cleaning ---
    if not args.skip_cleaning:
        print("--- Running Cleaning ---")
        run_cleaning(raw_dir=args.raw_dir, cleaned_dir=args.cleaned_dir)
        print("--- Cleaning completed successfully ---\n")

    # --- Step 3: Preparation ---
    if not args.skip_preparation:
        print("--- Running Preparation ---")
        run_preparation(cleaned_dir=args.cleaned_dir, processed_dir=args.processed_dir)
        print("--- Preparation completed successfully ---\n")

    # --- Step 4: Training ---
    if not args.skip_training:
        print("--- Running Training ---")
        # The training script already uses a config file, so we just need to ensure
        # it knows where to find the prepared data.
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)

            # Override data path to ensure it uses the output of the preparation step
            config.setdefault('data', {})['path'] = os.path.join(args.processed_dir, 'train.txt')

            run_training(config)
            print("--- Training completed successfully ---\n")
        except FileNotFoundError:
            print(f"Error: Training config file not found at '{args.config}'", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred during training: {e}", file=sys.stderr)

    print("Pipeline finished.")

    # --- Hub Synchronization (Post-run) ---
    if not args.no_sync:
        print("\n--- Synchronizing with Hugging Face Hub (Pushing) ---")
        try:
            subprocess.run(["python3", "scripts/sync_hub.py", "push", "--target", "all"], check=True)
            print("--- Synchronization (Push) complete ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not push to Hugging Face Hub. Local changes are not saved to the cloud. Error: {e}", file=sys.stderr)
    else:
        print("\n--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---")


if __name__ == "__main__":
    main()
