import argparse
import sys
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

def main():
    """
    Main function to run the miniature-memory data and training pipeline.

    This script orchestrates the validation, cleaning, preparation, and training
    stages. Each stage can be skipped via command-line arguments, and imports
    are deferred to reduce memory overhead and improve startup time.
    """
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="A unified script to run the miniature-memory pipeline."
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the raw data validation step."
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip the dataset cleaning step."
    )
    parser.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Skip the data preparation step."
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the model training step."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/small.yaml",
        help="Path to the training configuration file."
    )
    args = parser.parse_args()

    print("Starting the miniature-memory pipeline...\n")

    if not args.skip_validation:
        print("--- Running Validation ---")
        from scripts.validate_raw import run_validation
        run_validation("dataset/raw")
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        print("--- Running Cleaning ---")
        from scripts.clean_dataset import run_cleaning
        run_cleaning("dataset/raw", "dataset/cleaned")
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        print("--- Running Preparation ---")
        from scripts.prepare_data import run_preparation
        run_preparation("dataset/cleaned", "dataset/processed")
        print("--- Preparation completed successfully ---\n")

    if not args.skip_training:
        print("--- Running Training ---")
        import yaml
        from training.train import run_training

        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found at {args.config}", file=sys.stderr)
            sys.exit(1)

        run_training(config)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
