import os
import argparse
import sys

# Add project root to the Python path to allow importing from the processing package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.segment import Segmenter

# Define constants
OUTPUT_FILENAME = "train.txt"

def find_text_files(directory):
    """Finds all .txt files in a directory and returns a sorted list of paths."""
    text_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                text_files.append(os.path.join(root, filename))
    # Sort the list of files for deterministic behavior
    text_files.sort()
    return text_files

def run_preparation(cleaned_dir: str, processed_dir: str):
    """
    Concatenates all cleaned text files into a single training corpus,
    utilizing the modular Segmenter for structural markers.

    Args:
        cleaned_dir: Directory containing cleaned .txt files.
        processed_dir: Directory to save the final train.txt.
    """
    print(f"Starting modular data preparation from '{cleaned_dir}'...\n")

    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_files = find_text_files(cleaned_dir)
    segmenter = Segmenter()

    if not cleaned_files:
        print("No cleaned text files found. Nothing to prepare.")
        return

    output_filepath = os.path.join(processed_dir, OUTPUT_FILENAME)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for filepath in cleaned_files:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    # Add story markers modularly
                    marked_content = segmenter.add_story_markers(content)
                    outfile.write(marked_content)

                relative_path = os.path.relpath(filepath, cleaned_dir)
                print(f"[✅ APPENDED] {relative_path}")

        print(f"\nData preparation complete.")
        print(f"  Total files concatenated: {len(cleaned_files)}")
        print(f"  Output created at: {output_filepath}")

    except Exception as e:
        print(f"\n[❌ ERROR] An error occurred during data preparation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare cleaned data for training using modular Segmenter."
    )
    parser.add_argument(
        "--cleaned_dir",
        type=str,
        default="dataset/cleaned",
        help="The directory containing the cleaned text files."
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="dataset/processed",
        help="The directory where the final training corpus will be saved."
    )
    args = parser.parse_args()
    run_preparation(args.cleaned_dir, args.processed_dir)
