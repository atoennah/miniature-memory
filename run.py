# ⚡ Bolt Refactor: From Script to Strategy
#
# This file, once a procedural script, has been refactored into a strategic,
# object-oriented orchestrator. The `Pipeline` class represents a clear and
# testable workflow, transforming a sequence of shell-like commands into a
# robust, self-documenting system.
#
# The guiding principles of this refactor are:
# 1.  **Encapsulation:** The `Pipeline` class owns the logic and state of the
#     entire process, preventing variables from leaking into the global scope.
# 2.  **Clarity:** Each step of the pipeline (`validate`, `clean`, `prepare`,
#     `train`) is a distinct method with a clear purpose and explicit dependencies.
# 3.  **Extensibility:** Adding a new step to the pipeline is now as simple as
#     adding a new method, without disrupting the existing flow.
# 4.  **Self-Documentation:** The code's structure and docstrings are designed to
#     make the pipeline's intent obvious, reducing the cognitive load for new
#     developers.

import argparse
import sys
import subprocess
from importlib.metadata import PackageNotFoundError, version

class Pipeline:
    """
    Orchestrates the entire data processing and model training workflow.

    This class encapsulates the logic for each stage of the pipeline, from
    data validation to model training and synchronization with the Hugging

    Face Hub. It is configured via command-line arguments that allow
    skipping specific stages.
    """
    def __init__(self, args):
        """
        Initializes the Pipeline with command-line arguments.

        Args:
            args: An argparse.Namespace object containing the script's configuration.
        """
        self.args = args

    def run(self):
        """
        Executes the full pipeline in the correct order.
        """
        print("Starting the miniature-memory pipeline...\n")
        self.sync_hub("pull")

        if not self.args.skip_validation:
            self.run_validation()
        if not self.args.skip_cleaning:
            self.run_cleaning()
        if not self.args.skip_preparation:
            self.run_preparation()
        if not self.args.skip_training:
            self.run_training()

        self.sync_hub("push")
        print("Pipeline finished.")

    def run_validation(self):
        """Runs the raw data validation step."""
        print("--- Running Validation ---")
        from scripts.validate_raw import run_validation
        run_validation("dataset/raw")
        print("--- Validation completed successfully ---\n")

    def run_cleaning(self):
        """Runs the dataset cleaning step."""
        print("--- Running Cleaning ---")
        from scripts.clean_dataset import run_cleaning
        run_cleaning("dataset/raw", "dataset/cleaned")
        print("--- Cleaning completed successfully ---\n")

    def run_preparation(self):
        """Runs the data preparation step."""
        print("--- Running Preparation ---")
        from scripts.prepare_data import run_preparation
        run_preparation("dataset/cleaned", "dataset/processed")
        print("--- Preparation completed successfully ---\n")

    def run_training(self):
        """Runs the model training step."""
        print("--- Running Training ---")
        from training.train import main as run_training
        run_training()
        print("--- Training completed successfully ---\n")

    def sync_hub(self, direction: str):
        """
        Synchronizes the local repository with the Hugging Face Hub.

        Args:
            direction: The direction of synchronization ('pull' or 'push').
        """
        action = "Pulling" if direction == "pull" else "Pushing"
        print(f"--- Synchronizing with Hugging Face Hub ({action}) ---")
        try:
            subprocess.run(["python3", "scripts/sync_hub.py", direction, "--target", "all"], check=True)
            print(f"--- Synchronization ({action}) complete ---\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not {direction} Hugging Face Hub. Continuing with local state. Error: {e}\n", file=sys.stderr)

def check_dependencies():
    """
    Verifies that all required packages are installed.

    This function reads `requirements.txt`, checks if each package is
    available in the environment, and provides a clear, user-friendly error
    message if any dependencies are missing. This proactive check prevents
    cryptic `ImportError` exceptions at runtime.
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
        print("--- Missing Dependencies ---", file=sys.stderr)
        print("Error: The following required packages are not installed:", file=sys.stderr)
        for package in missing_packages:
            print(f"  - {package}", file=sys.stderr)
        print("\nPlease install the missing dependencies by running:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print("--------------------------", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main entry point for the script.

    Parses command-line arguments, checks for dependencies, and runs the pipeline.
    """
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

    # Pass any unknown arguments to the training script
    sys.argv = [sys.argv[0]] + unknown

    pipeline = Pipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
