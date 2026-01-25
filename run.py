import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

# --- Bolt Refactor: Direct Imports ---
# Instead of inefficiently calling scripts as separate processes, we now import
# their main functions directly. This creates a single, unified, and faster
# execution pipeline.
from scripts.sync_hub import pull_state, push_state
from scripts.validate_raw import run_validation
from scripts.clean_dataset import run_cleaning
from scripts.prepare_data import run_preparation
# --- End Bolt Refactor ---

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
    # --- Bolt Refactor: Define data directories centrally ---
    parser.add_argument(
        "--raw-data-dir", type=str, default="dataset/raw", help="Directory for raw data."
    )
    parser.add_argument(
        "--cleaned-data-dir", type=str, default="dataset/cleaned", help="Directory for cleaned data."
    )
    parser.add_argument(
        "--processed-data-dir", type=str, default="dataset/processed", help="Directory for processed data."
    )
    # --- End Bolt Refactor ---
    args, unknown = parser.parse_known_args()

    # At Startup (The Morning Briefing): Pull the latest state from the Hub.
    if not args.no_sync:
        print("--- Synchronizing with Hugging Face Hub (Pulling) ---")
        try:
            # --- Bolt Refactor: Direct function call ---
            pull_state(target="all")
            print("--- Synchronization (Pull) complete ---\n")
        except Exception as e:
            print(f"Warning: Could not pull from Hugging Face Hub. Continuing with local state. Error: {e}\n", file=sys.stderr)
    else:
        print("--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---\n")

    check_dependencies()

    # --- Bolt Refactor: Redundant sys.argv manipulation removed ---
    print("Starting the miniature-memory pipeline...\n")

    if not args.skip_validation:
        print("--- Running Validation ---")
        # --- Bolt Refactor: Direct function call ---
        run_validation(args.raw_data_dir)
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        print("--- Running Cleaning ---")
        # --- Bolt Refactor: Direct function call ---
        run_cleaning(args.raw_data_dir, args.cleaned_data_dir)
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        print("--- Running Preparation ---")
        # --- Bolt Refactor: Direct function call ---
        run_preparation(args.cleaned_data_dir, args.processed_data_dir)
        print("--- Preparation completed successfully ---\n")

    if not args.skip_training:
        print("--- Running Training ---")
        try:
            from training.train import main as run_training
            # Pass unknown args to the training script
            sys.argv = [sys.argv[0]] + unknown
            run_training()
        except ImportError:
            print("Error: Could not import the training module.", file=sys.stderr)
            sys.exit(1)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")

    # After Training (The Clock Out): Push the new state to the Hub, if not disabled.
    if not args.no_sync:
        print("\n--- Synchronizing with Hugging Face Hub (Pushing) ---")
        try:
            # --- Bolt Refactor: Direct function call ---
            push_state(target="all")
            print("--- Synchronization (Push) complete ---")
        except Exception as e:
            print(f"Warning: Could not push to Hugging Face Hub. Local changes are not saved to the cloud. Error: {e}", file=sys.stderr)
    else:
        print("\n--- Skipping Hugging Face Hub synchronization as per --no-sync flag ---")


if __name__ == "__main__":
    main()
