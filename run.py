import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

# Modular Imports from the data pipeline
from scripts.validate_raw import run_validation
from scripts.clean_dataset import run_cleaning
from scripts.prepare_data import run_preparation
from scripts.sync_hub import push_state, pull_state

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
    # --- Directory Arguments ---
    parser.add_argument("--raw-dir", type=str, default="dataset/raw", help="Directory for raw data.")
    parser.add_argument("--cleaned-dir", type=str, default="dataset/cleaned", help="Directory for cleaned data.")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Directory for processed data.")

    # --- Pipeline Control Arguments ---
    parser.add_argument("--skip-validation", action="store_true", help="Skip the raw data validation step.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip the dataset cleaning step.")
    parser.add_argument("--skip-preparation", action="store_true", help="Skip the data preparation step.")
    parser.add_argument("--skip-training", action="store_true", help="Skip the model training step.")
    parser.add_argument("--no-sync", action="store_true", help="Disable Hugging Face Hub synchronization.")

    # --- Training-Specific Arguments ---
    parser.add_argument("--config", type=str, default="training/configs/small.yaml", help="Path to the training configuration file.")
    args, unknown = parser.parse_known_args()

    # At Startup: Pull the latest state from the Hub.
    if not args.no_sync:
        print("--- Synchronizing with Hugging Face Hub (Pulling) ---")
        try:
            pull_state(target="all")
            print("--- Synchronization (Pull) complete ---\n")
        except (IOError, ValueError) as e:
            print(f"Warning: Could not pull from Hugging Face Hub. Continuing with local state. Error: {e}\n", file=sys.stderr)
    else:
        print("--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---\n")

    check_dependencies()

    # Pass any unknown arguments to the training script
    sys.argv = [sys.argv[0]] + unknown

    print("Starting the miniature-memory pipeline...\n")

    # --- Data Processing Pipeline ---
    if not args.skip_validation:
        print("--- Running Validation ---")
        run_validation(raw_data_dir=args.raw_dir)
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        print("--- Running Cleaning ---")
        run_cleaning(raw_dir=args.raw_dir, cleaned_dir=args.cleaned_dir)
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        print("--- Running Preparation ---")
        run_preparation(cleaned_dir=args.cleaned_dir, processed_dir=args.processed_dir)
        print("--- Preparation completed successfully ---\n")

    # --- Model Training ---
    if not args.skip_training:
        print("--- Running Training ---")
        try:
            # The training script is already designed to be importable
            from training.train import main as run_training
            run_training()
        except ImportError as e:
            print(f"Error: Could not import training module. Ensure it is correctly structured. Details: {e}", file=sys.stderr)
            sys.exit(1)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")

    # After Training: Push the new state to the Hub.
    if not args.no_sync:
        print("\n--- Synchronizing with Hugging Face Hub (Pushing) ---")
        try:
            push_state(target="all")
            print("--- Synchronization (Push) complete ---")
        except (IOError, ValueError) as e:
            print(f"Warning: Could not push to Hugging Face Hub. Local changes are not saved to the cloud. Error: {e}", file=sys.stderr)
    else:
        print("\n--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---")


if __name__ == "__main__":
    main()
