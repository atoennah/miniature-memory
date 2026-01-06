import argparse
import sys
import yaml


def main():
    """
    A unified script to run the miniature-memory pipeline.
    This script orchestrates the validation, cleaning, preparation, and training stages.
    """
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
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Defer the import of the training module
        from training.train import main as run_training
        run_training(config)
        print("--- Training completed successfully ---\n")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
