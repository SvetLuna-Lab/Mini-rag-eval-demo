"""
Lightweight document indexer for a tiny RAG demo.

- Loads .txt files from a directory into Document objects
- Uses TF–IDF + cosine similarity for retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """
    Simple text document used by the indexer and RAG pipeline.

    Attributes
    ----------
    doc_id : str
        Identifier (usually the filename without extension).
    text : str
        Raw text content.
    """

    doc_id: str
    text: str


def load_txt_corpus_from_dir(corpus_dir: Path) -> List[Document]:
    """
    Load all .txt files from `corpus_dir` into a list of Document objects.

    Parameters
    ----------
    corpus_dir : Path
        Directory that contains .txt files.

    Returns
    -------
    List[Document]
        List of loaded documents sorted by filename.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    docs: List[Document] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(doc_id=path.stem, text=text))

    if not docs:
        raise ValueError(f"No .txt files found in corpus directory: {corpus_dir}")

    return docs


class DocumentIndexer:
    """
    Tiny TF–IDF based indexer for text retrieval.

    This is intentionally simple: good enough for a demo while still easy
    to read and reason about.
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix = None  # type: ignore[assignment]
        self._docs: List[Document] = []

    @property
    def docs(self) -> List[Document]:
        """Return the underlying corpus."""
        return self._docs

    def fit(self, docs: Iterable[Document]) -> None:
        """
        Build an index over the given documents.

        Parameters
        ----------
        docs : Iterable[Document]
            Documents to index.
        """
        self._docs = list(docs)
        if not self._docs:
            raise ValueError("Cannot build index on an empty document list.")

        texts = [d.text for d in self._docs]

        self._vectorizer = TfidfVectorizer()
        self._doc_matrix = self._vectorizer.fit_transform(texts)

    def is_ready(self) -> bool:
        """Return True if the index was already built."""
        return self._vectorizer is not None and self._doc_matrix is not None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most similar documents for a query.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int, optional
            Number of documents to return, by default 3.

        Returns
        -------
        List[Tuple[Document, float]]
            A list of (Document, similarity_score) sorted from best to worst.
        """
        if not self.is_ready():
            raise RuntimeError("Index is not built. Call fit() first.")

        assert self._vectorizer is not None
        assert self._doc_matrix is not None

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._doc_matrix)[0]

        top_k = max(1, min(top_k, len(self._docs)))
        top_idx = np.argsort(sims)[::-1][:top_k]

        results: List[Tuple[Document, float]] = []
        for idx in top_idx:
            results.append((self._docs[int(idx)], float(sims[int(idx)])))

        return results
