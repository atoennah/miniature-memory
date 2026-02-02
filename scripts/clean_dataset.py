import os
import argparse
from processing.normalize import TextNormalizer
from processing.quality_filter import QualityFilter

def run_cleaning(raw_dir, cleaned_dir):
    """
    Walks the raw data directory, cleans files using modular components,
    and saves them to the cleaned directory.
    """
    print(f"Starting cleaning process from '{raw_dir}' to '{cleaned_dir}'...\n")

    normalizer = TextNormalizer()
    q_filter = QualityFilter()

    cleaned_count = 0
    skipped_count = 0

    for root, _, files in os.walk(raw_dir):
        for filename in files:
            if filename.endswith(".txt"):
                raw_filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(raw_filepath, raw_dir)
                cleaned_filepath = os.path.join(cleaned_dir, relative_path)

                # Ensure the destination directory exists
                os.makedirs(os.path.dirname(cleaned_filepath), exist_ok=True)

                try:
                    with open(raw_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # 1. Filter out paragraphs with noise
                    content = q_filter.filter_paragraphs(content)

                    # 2. Normalize text
                    content = normalizer.normalize(content)

                    # 3. Final validation (Indonesian language check)
                    if content and q_filter.validate(content):
                        with open(cleaned_filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"[✅ CLEANED] {relative_path}")
                        cleaned_count += 1
                    else:
                        print(f"[⚠️ SKIPPED] {relative_path} (failed quality/language filter)")
                        skipped_count += 1

                except Exception as e:
                    print(f"[❌ ERROR] Failed to process {relative_path}: {e}")
                    skipped_count += 1

    print("\nCleaning Complete.")
    print(f"  Cleaned: {cleaned_count}")
    print(f"  Skipped/Errors: {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean raw text files using modular processing components."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="dataset/raw",
        help="The directory containing the raw text files."
    )
    parser.add_argument(
        "--cleaned_dir",
        type=str,
        default="dataset/cleaned",
        help="The directory where cleaned text files will be saved."
    )
    args = parser.parse_args()
    run_cleaning(args.raw_dir, args.cleaned_dir)
