"""
Very small evaluation module for RAG answers.

The goal is not to be state-of-the-art, but to show how one *could*
attach metrics to a RAG system.

We implement token-level precision / recall / F1 and Jaccard overlap.
"""

from __future__ import annotations

from typing import Dict, Tuple


def _tokenize(text: str) -> set[str]:
    """
    Extremely naive tokenizer: lowercase and split on whitespace.

    In a real system you would use a proper tokenizer,
    but this is enough for a demo.
    """
    return {t for t in text.lower().split() if t}


def _precision_recall_f1(pred_tokens: set[str], ref_tokens: set[str]) -> Tuple[float, float, float]:
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    intersection = pred_tokens & ref_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def _jaccard(pred_tokens: set[str], ref_tokens: set[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    union = pred_tokens | ref_tokens
    if not union:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(union)


def evaluate_answer(predicted: str, reference: str) -> Dict[str, float]:
    """
    Compute simple lexical metrics between a predicted answer and a reference.

    Parameters
    ----------
    predicted : str
        Model (or stub) answer.
    reference : str
        Ground-truth / target answer.

    Returns
    -------
    Dict[str, float]
        Dictionary with metrics: precision, recall, f1, jaccard.
    """
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference)

    precision, recall, f1 = _precision_recall_f1(pred_tokens, ref_tokens)
    jaccard = _jaccard(pred_tokens, ref_tokens)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }
