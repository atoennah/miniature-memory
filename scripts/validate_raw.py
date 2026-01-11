import os
import argparse

# Define validation constants
MIN_FILE_LENGTH = 50  # Minimum number of characters
MIN_PRINTABLE_RATIO = 0.85 # Minimum ratio of printable characters

def is_printable(char):
    """Checks if a character is printable, including common whitespace."""
    return char.isprintable() or char in ('\n', '\r', '\t')

def validate_file(filepath):
    """
    Validates a single raw text file based on length and content.

    Args:
        filepath (str): The path to the text file.

    Returns:
        tuple: A tuple containing a boolean (True if valid) and a message.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if len(content) < MIN_FILE_LENGTH:
            return False, f"Too short ({len(content)} chars)"

        printable_chars = sum(1 for char in content if is_printable(char))
        printable_ratio = printable_chars / len(content) if len(content) > 0 else 0

        if printable_ratio < MIN_PRINTABLE_RATIO:
            return False, f"Low printable ratio ({printable_ratio:.2f})"

        return True, "Valid"

    except Exception as e:
        return False, f"Error reading file: {e}"

def run_validation(raw_data_dir):
    """
    Walks through the raw data directory and validates each text file.
    """
    print(f"Starting validation in '{raw_data_dir}'...\n")
    validated_count = 0
    failed_count = 0

    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                is_valid, message = validate_file(filepath)

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
        description="Validate raw text files in the dataset."
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="dataset/raw",
        help="The directory containing the raw text files.",
    )
    args = parser.parse_args()
    run_validation(args.raw_data_dir)
