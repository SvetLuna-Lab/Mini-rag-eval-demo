from src.eval import evaluate_answer


def test_evaluate_answer_basic_overlap() -> None:
    pred = "rocket engine health monitoring"
    ref = "health monitoring for rocket engines"

    metrics = evaluate_answer(pred, ref)

    assert 0.0 < metrics["precision"] <= 1.0
    assert 0.0 < metrics["recall"] <= 1.0
    assert 0.0 < metrics["f1"] <= 1.0
    assert 0.0 < metrics["jaccard"] <= 1.0
