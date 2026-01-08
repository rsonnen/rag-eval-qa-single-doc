#!/usr/bin/env python3
"""Generate and validate questions from a document.

CLI wrapper around the orchestrator module. Runs the full question generation
pipeline: generate -> deduplicate -> validate.

Usage:
    uv run python scripts/generate_questions.py \
        --document /path/to/document.pdf \
        --corpus /path/to/corpus/dir \
        --scenario rag_eval \
        --mode visual \
        --count 5 \
        --max-failures 5 \
        --output results.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from single_doc_generator.config import load_corpus_config
from single_doc_generator.models import GenerationMode, QAPair, RejectedQA
from single_doc_generator.orchestrator import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    run_generation,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and validate questions from a document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--document",
        type=Path,
        required=True,
        help="Path to the document (PDF, XML, or text file)",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus directory containing corpus.yaml",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name from corpus.yaml (e.g., rag_eval, graduate_exam)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["textual", "visual"],
        default="textual",
        help="Generation mode: textual (text only) or visual (visual content)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of questions to generate (default: 5)",
    )

    parser.add_argument(
        "--max-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=f"Consecutive failures before exhaustion (default: {DEFAULT_MAX_CONSECUTIVE_FAILURES})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results (optional, prints to stdout if not specified)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Run the question generation pipeline.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    import logging

    # Configure logging - always show INFO for the script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Only enable DEBUG for our package, not third-party libraries
    if args.verbose:
        logging.getLogger("single_doc_generator").setLevel(logging.DEBUG)

    if not args.document.exists():
        print(f"Error: Document not found: {args.document}", file=sys.stderr)
        return 1

    if not args.corpus.exists():
        print(f"Error: Corpus directory not found: {args.corpus}", file=sys.stderr)
        return 1

    corpus_yaml = args.corpus / "corpus.yaml"
    if not corpus_yaml.exists():
        print(f"Error: corpus.yaml not found in {args.corpus}", file=sys.stderr)
        return 1

    try:
        corpus_config = load_corpus_config(args.corpus)
    except Exception as e:
        print(f"Error loading corpus config: {e}", file=sys.stderr)
        return 1

    if args.scenario not in corpus_config.scenarios:
        available = ", ".join(corpus_config.scenarios.keys())
        print(
            f"Error: Scenario '{args.scenario}' not found. Available: {available}",
            file=sys.stderr,
        )
        return 1

    mode = GenerationMode.VISUAL if args.mode == "visual" else GenerationMode.TEXTUAL

    print(f"Document: {args.document.name}")
    print(f"Corpus: {corpus_config.name}")
    print(f"Scenario: {args.scenario}")
    print(f"Mode: {args.mode}")
    print(f"Target: {args.count} questions")
    print(f"Max consecutive failures: {args.max_failures}\n")

    def on_attempt(accepted: int, target: int, attempt: int) -> None:
        print(f"[{accepted}/{target}] Attempt {attempt}...")

    def on_accepted(qa: QAPair) -> None:
        print(f"  ACCEPTED: {qa.question[:60]}...")

    def on_rejected(rejected: RejectedQA) -> None:
        print(f"  REJECTED ({rejected.rejection_reason.value}): {rejected.question[:50]}...")

    def on_exhausted(reason: str) -> None:
        print(f"  -> Exhausted: {reason}")

    result = run_generation(
        document_path=args.document,
        corpus_path=args.corpus,
        corpus_config=corpus_config,
        scenario_name=args.scenario,
        mode=mode,
        target_count=args.count,
        max_consecutive_failures=args.max_failures,
        on_attempt=on_attempt,
        on_accepted=on_accepted,
        on_rejected=on_rejected,
        on_exhausted=on_exhausted,
    )

    # Set timestamp
    result.timestamp = datetime.now(timezone.utc).isoformat()

    output_dict = result.model_dump(mode="json")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("RESULTS (JSON)")
        print("=" * 60)
        print(json.dumps(output_dict, indent=2))

    stats = result.stats
    print(f"\nSummary: {stats.accepted_count}/{args.count} questions accepted")
    print(f"Total attempts: {stats.total_attempts}")
    print(f"Rejected: {stats.rejected_count}")
    if stats.rejection_breakdown:
        print("Rejection breakdown:")
        for reason, count in sorted(stats.rejection_breakdown.items()):
            print(f"  {reason}: {count}")
    if stats.exhausted:
        print(f"Exhaustion: {stats.exhaustion_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
