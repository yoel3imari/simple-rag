import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from src.config import INGEST_STATE_PATH, SUPPORTED_EXTENSIONS
from src.constants import TransformMode
from src.crawl import crawl_site
from src.ingest import ingest
from src.query import query


def add_eval_parser(subparsers):
    from src.eval import run_eval, print_report

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval & generation quality")
    eval_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder re-ranking",
    )
    eval_parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable BM25 keyword search (dense-only hybrid)",
    )
    eval_parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Only dense vector search (no BM25, no rerank)",
    )
    eval_parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Include faithfulness evaluation (extra LLM calls)",
    )
    eval_parser.add_argument(
        "--transform",
        choices=[m.value for m in TransformMode],
        default=None,
        help="Query transformation mode (default: config QUERY_TRANSFORM_MODE)",
    )
    eval_parser.add_argument(
        "--compare-transforms",
        action="store_true",
        help="Run eval across all transform modes and print a comparison table",
    )


def handle_eval(args):
    from src.eval import run_eval, print_report, compare_transforms

    if args.compare_transforms:
        compare_transforms()
        return

    use_rerank = not args.no_rerank and not args.dense_only
    use_bm25 = not args.no_bm25 and not args.dense_only

    summary = run_eval(
        use_rerank=use_rerank,
        use_bm25=use_bm25,
        faithfulness=args.faithfulness,
        transform_mode=args.transform,
    )
    print_report(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG system")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text document or web URL")
    ingest_parser.add_argument(
        "file", nargs="*", help="One or more files, directories, or URLs to ingest"
    )
    ingest_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the vector database from scratch",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if already ingested",
    )
    ingest_parser.add_argument(
        "--status",
        action="store_true",
        help="Print ingest state (sources + chunk counts) and exit",
    )

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument(
        "--transform",
        choices=[m.value for m in TransformMode],
        default=None,
        help="Query transformation mode (default: config QUERY_TRANSFORM_MODE)",
    )

    add_eval_parser(subparsers)

    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website and ingest all pages")
    crawl_parser.add_argument("url", help="Starting URL for the crawl")
    crawl_parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to crawl")
    crawl_parser.add_argument("--max-depth", type=int, default=2, help="Maximum crawl depth")
    crawl_parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    crawl_parser.add_argument("--scraper", choices=["crawl4ai", "jina"], default="crawl4ai", help="Scraping backend")
    crawl_parser.add_argument("--force", action="store_true", help="Re-crawl already ingested URLs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "ingest":
        if args.status:
            if INGEST_STATE_PATH.exists():
                state = json.loads(INGEST_STATE_PATH.read_text())
                print(json.dumps(state, indent=2))
            else:
                print("no ingest state yet")
            return

        if not args.file:
            ingest_parser.error("the following arguments are required: file")

        expanded: list[str] = []
        for p in args.file:
            path = Path(p)
            if path.is_dir():
                expanded.extend(
                    sorted(
                        str(f)
                        for f in path.rglob("*")
                        if f.suffix.lower() in SUPPORTED_EXTENSIONS
                    )
                )
            else:
                expanded.append(p)
        ingest(expanded, recreate=args.recreate, force=args.force)
    elif args.command == "query":
        answer, sources = query(args.question, transform_mode=args.transform)
        print(f"\n=== Answer ===\n{answer}")
        print(f"\n=== Sources ===\n{sources}")
    elif args.command == "crawl":
        pages = crawl_site(
            args.url,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            delay=args.delay,
            scraper=args.scraper,
        )
        urls = [p["url"] for p in pages]
        print(f"\n[crawl] Crawled {len(pages)} pages, now ingesting...")
        ingest(urls, recreate=False, force=args.force)
    elif args.command == "eval":
        handle_eval(args)


if __name__ == "__main__":
    main()
