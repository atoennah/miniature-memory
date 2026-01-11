import argparse
import sys
import subprocess
from importlib.metadata import PackageNotFoundError, version


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

def _handle_import_error(module_name):
    """Prints a user-friendly error message and exits."""
    print(f"Error: The required module '{module_name}' is not installed.", file=sys.stderr)
    print("Please install the necessary dependencies by running:", file=sys.stderr)
    print("  pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

def main():
    """
    Main function to run the miniature-memory data and training pipeline.

    This script orchestrates the validation, cleaning, preparation, and training
    stages. Each stage can be skipped via command-line arguments.
    """
    # At Startup (The Morning Briefing): Pull the latest state from the Hub.
    print("--- Synchronizing with Hugging Face Hub (Pulling) ---")
    try:
        subprocess.run(["python3", "scripts/sync_hub.py", "pull", "--target", "all"], check=True)
        print("--- Synchronization (Pull) complete ---\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not pull from Hugging Face Hub. Continuing with local state. Error: {e}\n", file=sys.stderr)

    check_dependencies()

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
    args, unknown = parser.parse_known_args()

    # Pass the unknown arguments to the training script
    sys.argv = [sys.argv[0]] + unknown

    print("Starting the miniature-memory pipeline...\n")

    if not args.skip_validation:
        try:
            from scripts.validate_raw import run_validation
        except ImportError:
            _handle_import_error("scripts.validate_raw")
        print("--- Running Validation ---")
        run_validation("dataset/raw")
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        try:
            from scripts.clean_dataset import run_cleaning
        except ImportError:
            _handle_import_error("scripts.clean_dataset")
        print("--- Running Cleaning ---")
        run_cleaning("dataset/raw", "dataset/cleaned")
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        try:
            from scripts.prepare_data import run_preparation
        except ImportError:
            _handle_import_error("scripts.prepare_data")
        print("--- Running Preparation ---")
        run_preparation("dataset/cleaned", "dataset/processed")
        print("--- Preparation completed successfully ---\n")

    if not args.skip_training:
        try:
            from training.train import main as run_training
        except ImportError:
            _handle_import_error("training.train")
        print("--- Running Training ---")
        run_training()
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")

    # After Training (The Clock Out): Push the new state to the Hub.
    print("\n--- Synchronizing with Hugging Face Hub (Pushing) ---")
    try:
        subprocess.run(["python3", "scripts/sync_hub.py", "push", "--target", "all"], check=True)
        print("--- Synchronization (Push) complete ---")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not push to Hugging Face Hub. Local changes are not saved to the cloud. Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
