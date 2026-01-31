import os
import argparse
import sys

# Add project root to the Python path to allow importing from the processing package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.quality_filter import QualityFilter

def run_validation(raw_data_dir: str):
    """
    Walks through the raw data directory and validates each text file
    using the modular QualityFilter.

    Args:
        raw_data_dir: Source directory with raw .txt files.
    """
    print(f"Starting modular validation in '{raw_data_dir}'...\n")

    q_filter = QualityFilter()
    validated_count = 0
    failed_count = 0

    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    is_valid, message = q_filter.validate_content(content)
                except Exception as e:
                    is_valid, message = False, str(e)

                if is_valid:
                    status = "✅ PASSED"
                    validated_count += 1
                else:
                    status = f"❌ FAILED: {message}"
                    failed_count += 1

                relative_path = os.path.relpath(filepath, raw_data_dir)
                print(f"[{status}] {relative_path}")

    print(f"\nValidation Complete.")
    print(f"  Passed: {validated_count}")
    print(f"  Failed: {failed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate raw text files in the dataset using modular QualityFilter."
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="dataset/raw",
        help="The directory containing the raw text files.",
    )
    args = parser.parse_args()
    run_validation(args.raw_data_dir)
