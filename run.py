import argparse
import sys
import subprocess
from importlib.metadata import PackageNotFoundError, version

import yaml
import os

# Directly import the main functions from the pipeline scripts
from scripts.validate_raw import run_validation
from scripts.clean_dataset import run_cleaning
from scripts.prepare_data import run_preparation
from training.train import run_training

def check_dependencies():
    """Checks if all the required packages are installed."""
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

def main():
    """
    Main function to run the miniature-memory data and training pipeline.

    This script orchestrates the validation, cleaning, preparation, and training
    stages. Each stage can be skipped via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A unified script to run the miniature-memory pipeline."
    )
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
        "--config",
        type=str,
        default="training/configs/small.yaml",
        help="Path to the training configuration file."
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Disable Hugging Face Hub synchronization."
    )
    # Add arguments for data directories to maintain script compatibility
    parser.add_argument(
        "--raw-dir", type=str, default="dataset/raw", help="Directory for raw data."
    )
    parser.add_argument(
        "--cleaned-dir", type=str, default="dataset/cleaned", help="Directory for cleaned data."
    )
    parser.add_argument(
        "--processed-dir", type=str, default="dataset/processed", help="Directory for processed data."
    )
    args, unknown = parser.parse_known_args()

    # At Startup (The Morning Briefing): Pull the latest state from the Hub.
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

    if not args.skip_validation:
        print("--- Running Validation ---")
        run_validation(args.raw_dir)
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        print("--- Running Cleaning ---")
        run_cleaning(args.raw_dir, args.cleaned_dir)
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        print("--- Running Preparation ---")
        run_preparation(args.cleaned_dir, args.processed_dir)
        print("--- Preparation completed successfully ---\n")

    if not args.skip_training:
        print("--- Running Training ---")
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Training configuration file not found at '{args.config}'", file=sys.stderr)
            sys.exit(1)

        # Override the data path in the config with the one from our arguments
        config.setdefault('data', {})['path'] = os.path.join(args.processed_dir, 'train.txt')

        # Pass the fully constructed config to the training function
        run_training(config)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")

    # After Training (The Clock Out): Push the new state to the Hub, if not disabled.
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
