import os
import re
import argparse

# Define cleaning constants
# Whitelist of characters: allows letters, numbers, basic punctuation, and whitespace.
ALLOWED_CHARS = re.compile(r'[^a-zA-Z0-9\s.,?!\'"()-]')
REPEATED_WHITESPACE = re.compile(r'[ \t]+')
REPEATED_NEWLINES = re.compile(r'\n{3,}')

# Blacklist for common Indonesian gambling ad keywords.
# These are case-insensitive.
GAMBLING_AD_KEYWORDS = [
    "slot gacor", "judi online", "daftar segera", "bonus new member",
    "zeus", "pragmatic play", "agen bola", "togel"
]

def clean_content(content):
    """
    Cleans the text content by removing non-allowed characters, normalizing
    whitespace, and filtering out paragraphs containing blacklisted keywords.

    Args:
        content (str): The raw text content.

    Returns:
        str: The cleaned text content, or an empty string if all content is filtered.
    """
    # --- "Slot Gacor" Filter ---
    # Split the content into paragraphs and filter out those containing ads.
    paragraphs = content.split('\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        # Check if any blacklisted keyword appears in the paragraph (case-insensitive)
        if not any(keyword in p.lower() for keyword in GAMBLING_AD_KEYWORDS):
            cleaned_paragraphs.append(p)

    # Re-join the content after filtering
    content = "\n".join(cleaned_paragraphs)

    # --- Standard Cleaning ---
    # Remove any characters not in our whitelist
    cleaned = ALLOWED_CHARS.sub('', content)

    # Normalize whitespace: replace tabs and multiple spaces with a single space
    cleaned = REPEATED_WHITESPACE.sub(' ', cleaned)

    # Reduce sequences of 3 or more newlines to 2
    cleaned = REPEATED_NEWLINES.sub('\n\n', cleaned)

    # Strip leading/trailing whitespace from the whole text
    cleaned = cleaned.strip()

    return cleaned

def run_cleaning(raw_dir, cleaned_dir):
    """
    Walks the raw data directory, cleans files, and saves them to the cleaned directory.
    """
    print(f"Starting cleaning process from '{raw_dir}' to '{cleaned_dir}'...\n")
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

                    cleaned_content = clean_content(raw_content)

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
        description="Clean raw text files and save them to a new directory."
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
