from pathlib import Path

from src.indexer import Document, DocumentIndexer


def test_indexer_fit_and_retrieve(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "docs"
    corpus_dir.mkdir()

    (corpus_dir / "a.txt").write_text("rocket engine health monitoring", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("customer churn prediction with ML", encoding="utf-8")

    docs = [
        Document(doc_id="a", text=(corpus_dir / "a.txt").read_text(encoding="utf-8")),
        Document(doc_id="b", text=(corpus_dir / "b.txt").read_text(encoding="utf-8")),
    ]

    indexer = DocumentIndexer()
    indexer.fit(docs)

    results = indexer.retrieve("rocket engine", top_k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.doc_id == "a"
    assert score > 0.0
