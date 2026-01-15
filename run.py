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

class Pipeline:
    """Orchestrates the data processing and model training pipeline."""

    def __init__(self, args):
        self.args = args

    def _run_validation(self):
        """Runs the raw data validation step."""
        if not self.args.skip_validation:
            try:
                from scripts.validate_raw import run_validation
            except ImportError:
                _handle_import_error("scripts.validate_raw")
            print("--- Running Validation ---")
            run_validation("dataset/raw")
            print("--- Validation completed successfully ---\n")

    def _run_cleaning(self):
        """Runs the dataset cleaning step."""
        if not self.args.skip_cleaning:
            try:
                from scripts.clean_dataset import run_cleaning
            except ImportError:
                _handle_import_error("scripts.clean_dataset")
            print("--- Running Cleaning ---")
            run_cleaning("dataset/raw", "dataset/cleaned")
            print("--- Cleaning completed successfully ---\n")

    def _run_preparation(self):
        """Runs the data preparation step."""
        if not self.args.skip_preparation:
            try:
                from scripts.prepare_data import run_preparation
            except ImportError:
                _handle_import_error("scripts.prepare_data")
            print("--- Running Preparation ---")
            run_preparation("dataset/cleaned", "dataset/processed")
            print("--- Preparation completed successfully ---\n")

    def _run_training(self):
        """Runs the model training step."""
        if not self.args.skip_training:
            try:
                from training.train import main as run_training
            except ImportError:
                _handle_import_error("training.train")
            print("--- Running Training ---")
            run_training()
            print("--- Training completed successfully ---\n")

    def run(self):
        """Executes the entire pipeline in the correct order."""
        print("Starting the miniature-memory pipeline...\n")
        self._run_validation()
        self._run_cleaning()
        self._run_preparation()
        self._run_training()
        print("Pipeline finished.")

def main():
    """
    Main function to configure and run the miniature-memory pipeline.
    """
    # At Startup: Pull the latest state from the Hub.
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

    # Pass unknown arguments to the training script
    sys.argv = [sys.argv[0]] + unknown

    pipeline = Pipeline(args)
    pipeline.run()

    # After Training: Push the new state to the Hub.
    print("\n--- Synchronizing with Hugging Face Hub (Pushing) ---")
    try:
        subprocess.run(["python3", "scripts/sync_hub.py", "push", "--target", "all"], check=True)
        print("--- Synchronization (Push) complete ---")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not push to Hugging Face Hub. Local changes are not saved to the cloud. Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
