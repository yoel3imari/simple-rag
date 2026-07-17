import argparse

from src.ingest import ingest
from src.query import query


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text document")
    ingest_parser.add_argument("file", help="Path to the text file")
    ingest_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the vector database from scratch",
    )

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.file, recreate=args.recreate)
    elif args.command == "query":
        answer, sources = query(args.question)
        print(f"\n=== Answer ===\n{answer}")
        print(f"\n=== Sources ===\n{sources}")


if __name__ == "__main__":
    main()
