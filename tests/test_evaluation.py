"""Smoke tests for the evaluation utilities."""

from __future__ import annotations

from student.evaluation import evaluate, recall_for_question
from student.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def _src(p: str, a: int, b: int) -> MinimalSource:
    return MinimalSource(file_path=p, first_character_index=a, last_character_index=b)


def test_recall_perfect() -> None:
    truths = [_src("a.py", 0, 100)]
    retrieved = [_src("a.py", 0, 100)]
    assert recall_for_question(retrieved, truths, k=1) == 1.0


def test_recall_partial_overlap() -> None:
    truths = [_src("a.py", 0, 100)]
    retrieved = [_src("a.py", 90, 200)]  # 10 chars overlap = 10% > 5%
    assert recall_for_question(retrieved, truths, k=1) == 1.0


def test_recall_no_overlap() -> None:
    truths = [_src("a.py", 0, 100)]
    retrieved = [_src("b.py", 0, 100)]
    assert recall_for_question(retrieved, truths, k=1) == 0.0


def test_evaluate_aggregates() -> None:
    truth = [
        AnsweredQuestion(
            question_id="q1", question="?", sources=[_src("a.py", 0, 100)], answer=""
        )
    ]
    student = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="?",
                retrieved_sources=[_src("a.py", 0, 100)],
            )
        ],
        k=5,
    )
    report = evaluate(student, truth, ks=(1, 5))
    assert report.recalls[1] == 1.0
    assert report.recalls[5] == 1.0
    assert report.questions_evaluated == 1
