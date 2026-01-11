import argparse
import sys
import yaml


def main():
    """
    A unified script to run the miniature-memory pipeline.
    This script orchestrates the validation, cleaning, preparation, and training stages.
    """
from importlib.metadata import PackageNotFoundError, version


from scripts.validate_raw import run_validation
from scripts.clean_dataset import run_cleaning
from scripts.prepare_data import run_preparation
from training.train import main as run_training

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
    # Deferring imports into conditional blocks to improve script startup time.
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
        from scripts.validate_raw import run_validation
        # Defer import to improve startup speed when skipping pipeline stages.
        from scripts.validate_raw import run_validation
        try:
            from scripts.validate_raw import run_validation
        except ImportError:
            _handle_import_error("scripts.validate_raw")
        print("--- Running Validation ---")
        from scripts.validate_raw import run_validation
        run_validation("dataset/raw")
        print("--- Validation completed successfully ---\n")

    if not args.skip_cleaning:
        from scripts.clean_dataset import run_cleaning
        # Defer import to improve startup speed when skipping pipeline stages.
        from scripts.clean_dataset import run_cleaning
        try:
            from scripts.clean_dataset import run_cleaning
        except ImportError:
            _handle_import_error("scripts.clean_dataset")
        print("--- Running Cleaning ---")
        from scripts.clean_dataset import run_cleaning
        run_cleaning("dataset/raw", "dataset/cleaned")
        print("--- Cleaning completed successfully ---\n")

    if not args.skip_preparation:
        from scripts.prepare_data import run_preparation
        # Defer import to improve startup speed when skipping pipeline stages.
        from scripts.prepare_data import run_preparation
        try:
            from scripts.prepare_data import run_preparation
        except ImportError:
            _handle_import_error("scripts.prepare_data")
        print("--- Running Preparation ---")
        from scripts.prepare_data import run_preparation
        run_preparation("dataset/cleaned", "dataset/processed")
        print("--- Preparation completed successfully ---\n")

    if not args.skip_training:
        from training.train import run_training
        # Defer import to improve startup speed when skipping pipeline stages.
        from training.train import run_training
        try:
            from training.train import run_training
            import yaml
        except ImportError as e:
            # Differentiate between our code and third-party libs
            if e.name == 'yaml':
                _handle_import_error('pyyaml')
            else:
                _handle_import_error("training.train")

        print("--- Running Training ---")
        from training.train import run_training
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        import yaml
        from training.train import run_training

        from training.train import run_training
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found at {args.config}", file=sys.stderr)
            print(f"Error: Configuration file not found at '{args.config}'", file=sys.stderr)
            sys.exit(1)

        # Re-apply the hardcoded values from the original train.py for consistency
        config.setdefault('training', {})['max_steps'] = 100
        config.setdefault('training', {})['eval_interval'] = 10
        config.setdefault('training', {})['output_dir'] = 'training/checkpoints'
        config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'
        config.setdefault('model', {})['block_size'] = 256

        # Defer the import of the training module
        from training.train import main as run_training
        run_training(config)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
