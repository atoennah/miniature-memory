
import os
import sys
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing import CleaningPipeline

# Blacklist for common Indonesian gambling ad keywords.
GAMBLING_AD_KEYWORDS = [
    "slot gacor", "judi online", "daftar segera", "bonus new member",
    "zeus", "pragmatic play", "agen bola", "togel"
]

def run_cleaning(raw_dir, cleaned_dir):
    """
    Walks the raw data directory, cleans files using the modular CleaningPipeline,
    and saves them to the cleaned directory.
    """
    print(f"Starting modular cleaning process from '{raw_dir}' to '{cleaned_dir}'...\n")
    pipeline = CleaningPipeline(blacklist_keywords=GAMBLING_AD_KEYWORDS)
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
                        raw_content = f.read()

                    cleaned_content = pipeline.process(raw_content)

                    if cleaned_content:
                        with open(cleaned_filepath, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        print(f"[✅ CLEANED] {relative_path}")
                        cleaned_count += 1
                    else:
                        print(f"[⚠️ SKIPPED] {relative_path} (empty after cleaning)")
                        skipped_count += 1

                except Exception as e:
                    print(f"[❌ ERROR] Failed to process {relative_path}: {e}")
                    skipped_count += 1

    print("\nCleaning Complete.")
    print(f"  Cleaned: {cleaned_count}")
    print(f"  Skipped/Errors: {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modularly clean raw text files and save them to a new directory."
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
