import os
import argparse
from collections import Counter
import glob

# Define Indonesian word lists for language register analysis
FORMAL_WORDS = {"saya", "anda", "bapak", "ibu", "tidak", "terima kasih", "silakan", "mohon"}
INFORMAL_WORDS = {"aku", "gue", "gw", "elo", "elu", "lu", "nggak", "gak", "makasih", "please", "plis"}

def count_words(text, word_set):
    """Counts occurrences of words from a given set in the text."""
    # Simple word tokenization
    words = text.lower().split()
    return sum(1 for word in words if word in word_set)

def analyze_file(filepath):
    """Analyzes a single text file for various statistics."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    return {
        "filepath": filepath,
        "char_count": len(content),
        "word_count": len(content.split()),
        "formal_word_count": count_words(content, FORMAL_WORDS),
        "informal_word_count": count_words(content, INFORMAL_WORDS),
    }

def run_stats(directory):
    """
    Calculates and displays statistics for all .txt files in a directory.
    """
    print(f"Calculating statistics for text files in '{directory}'...\n")

    filepaths = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    if not filepaths:
        print("No text files found.")
        return

    total_char_count = 0
    total_word_count = 0
    total_formal_words = 0
    total_informal_words = 0

    for filepath in filepaths:
        stats = analyze_file(filepath)
        if "error" in stats:
            print(f"[⚠️ SKIPPED] {filepath} (Error: {stats['error']})")
            continue

        total_char_count += stats["char_count"]
        total_word_count += stats["word_count"]
        total_formal_words += stats["formal_word_count"]
        total_informal_words += stats["informal_word_count"]

    print("--- Dataset Statistics ---")
    print(f"  Total Files: {len(filepaths)}")
    print(f"  Total Characters: {total_char_count:,}")
    print(f"  Total Words: {total_word_count:,}")
    print("\n--- Language Register Analysis ---")
    print(f"  Formal Word Occurrences: {total_formal_words:,}")
    print(f"  Informal Word Occurrences: {total_informal_words:,}")

    if total_formal_words + total_informal_words > 0:
        formal_ratio = total_formal_words / (total_formal_words + total_informal_words) * 100
        informal_ratio = total_informal_words / (total_formal_words + total_informal_words) * 100
        print(f"\n  Register Ratio:")
        print(f"    - Formal: {formal_ratio:.2f}%")
        print(f"    - Informal: {informal_ratio:.2f}%")
    else:
        print("\n  Register Ratio: Not enough data for analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate statistics for a dataset of text files."
    )
    parser.add_argument(
        "directory",
        type=str,
        default="dataset/cleaned",
        nargs='?',
        help="The directory to analyze (default: dataset/cleaned)."
    )
    args = parser.parse_args()
    run_stats(args.directory)
