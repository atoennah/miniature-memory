import argparse
import sys
import os

# Add the project root to the Python path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scraper.commands.search import run_search
from scraper.commands.process import run_process

def main():
    """
    The main entry point for the command-line interface.
    This function parses arguments and dispatches to the appropriate command module.
    """
    parser = argparse.ArgumentParser(
        description="A CLI for the auto-scraping and training pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search command - delegates to the search module
    search_parser = subparsers.add_parser(
        "search", help="Discover URLs for a given query."
    )
    search_parser.add_argument("query", type=str, help="The search query.")
    search_parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="The number of results to return."
    )
    search_parser.set_defaults(func=run_search)

    # Process command - delegates to the process module
    process_parser = subparsers.add_parser(
        "process", help="Heuristically crawl index pages to find and process story URLs."
    )
    process_parser.set_defaults(func=run_process)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        # Execute the function associated with the chosen command
        args.func(args)

if __name__ == "__main__":
    main()
