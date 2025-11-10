"""
Command-line interface for the mini RAG quality demo.

Usage examples
--------------

Index all docs in ./docs and run interactive mode:

    python -m src.cli --corpus-dir docs

Run a single question with an optional reference answer:

    python -m src.cli --corpus-dir docs \
        --question "What is this project about?" \
        --reference "This is a small RAG quality demo."

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .eval import evaluate_answer
from .indexer import DocumentIndexer, load_txt_corpus_from_dir
from .rag_pipeline import RagPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini RAG quality demo: index, retrieve, generate, evaluate."
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="docs",
        help="Directory with .txt files to index (default: docs/).",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Single question to ask. If omitted, runs an interactive REPL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Optional reference answer used to compute simple lexical metrics.",
    )
    return parser.parse_args()


def build_pipeline(corpus_dir: Path) -> RagPipeline:
    """
    Helper to load documents, build an indexer and create the RAG pipeline.
    """
    docs = load_txt_corpus_from_dir(corpus_dir)
    indexer = DocumentIndexer()
    indexer.fit(docs)
    return RagPipeline(indexer=indexer)


def run_single_question(
    pipeline: RagPipeline,
    question: str,
    reference: Optional[str],
    top_k: int = 3,
) -> None:
    """
    Run the pipeline for one question and optionally print evaluation metrics.
    """
    result = pipeline.run(question, top_k=top_k)

    print("\n=== RAG RESULT ===")
    print(f"Question: {result.question}\n")
    print("Answer:")
    print(result.answer)
    print("\nRetrieved contexts:")
    for doc, score in zip(result.contexts, result.scores):
        print(f"- {doc.doc_id} (score={score:.3f})")

    if reference is not None:
        metrics = evaluate_answer(result.answer, reference)
        print("\nEvaluation against reference:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.3f}")


def interactive_loop(pipeline: RagPipeline, top_k: int = 3) -> None:
    """
    Simple REPL for querying the RAG pipeline from the terminal.
    """
    print("Mini RAG quality demo")
    print("Type a question and press Enter.")
    print("Press Ctrl+C or send an empty line to exit.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            print("Bye!")
            break

        run_single_question(pipeline, question, reference=None, top_k=top_k)


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)

    pipeline = build_pipeline(corpus_dir)

    if args.question:
        run_single_question(
            pipeline,
            question=args.question,
            reference=args.reference,
            top_k=args.top_k,
        )
    else:
        interactive_loop(pipeline, top_k=args.top_k)


if __name__ == "__main__":
    main()
