# Mini-rag-eval-demo

Small, educational project that shows how to build a **minimal RAG pipeline**
with a **simple evaluation loop**.

It is not about state-of-the-art models, but about clean structure:

- tiny document indexer (TF–IDF + cosine similarity),
- minimal RAG pipeline (retrieve + simple LLM stub),
- toy lexical metrics (precision / recall / F1 / Jaccard),
- a couple of tests and a clear folder layout.

---

## Project structure

```text
mini-rag-eval-demo/
├─ docs/
│  ├─ .gitkeep
│  ├─ doc1.txt
│  ├─ doc2.txt
│  ├─ doc3.txt
│  └─ doc4.txt
├─ src/
│  ├─ __init__.py
│  ├─ indexer.py
│  ├─ rag_pipeline.py
│  ├─ eval.py
│  └─ cli.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_indexer.py
│  └─ test_eval.py
├─ requirements.txt
├─ .gitignore
└─ README.md


## Corpus (docs/)

The `docs/` folder contains a tiny text corpus focused on RAG and evaluation:

- `doc1.txt` – short overview of this demo project
- `doc2.txt` – what RAG is and why retrieval matters
- `doc3.txt` – simple lexical metrics (precision, recall, F1, Jaccard)
- `doc4.txt` – why the project is intentionally small and transparent


Installation

python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt


(Optional) If you want to run tests:

pip install pytest
pytest


Usage

Index documents from docs/ and run an interactive QA loop:

python -m src.cli --corpus-dir docs


Run a single question with an optional reference answer,
so that simple metrics are printed:

python -m src.cli \
  --corpus-dir docs \
  --question "What is this project about?" \
  --reference "It is a small RAG evaluation demo with TF-IDF indexer."


You will see:

the generated answer from the SimpleLLMStub,

which documents were retrieved and their scores,

lexical metrics (precision / recall / F1 / Jaccard) against the reference.


Why this project exists

For real-world systems, model is only part of the story.
The “patient” is the whole infrastructure + data volume.

This repo focuses on:

clear, inspectable components instead of black boxes,

making evaluation explicit even on a tiny toy example,

showing that good engineering habits scale from small demos to larger systems.

