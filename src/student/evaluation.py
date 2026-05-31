

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from student.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def _overlap(a: MinimalSource, b: MinimalSource) -> int:
    """Number of overlapping characters between two sources on the same file."""
    if a.file_path != b.file_path:
        return 0
    lo = max(a.first_character_index, b.first_character_index)
    hi = min(a.last_character_index, b.last_character_index)
    return max(0, hi - lo)


def _source_len(s: MinimalSource) -> int:
    return max(0, s.last_character_index - s.first_character_index)


def is_found(
    retrieved: Sequence[MinimalSource],
    truth: MinimalSource,
    min_overlap_ratio: float,
) -> bool:
    """A truth source is "found" if any retrieved source overlaps it enough."""
    truth_len = _source_len(truth)
    if truth_len == 0:
        return any(
            r.file_path == truth.file_path
            and r.first_character_index <= truth.first_character_index
            and r.last_character_index >= truth.last_character_index
            for r in retrieved
        )
    threshold = max(1, int(truth_len * min_overlap_ratio))
    return any(_overlap(r, truth) >= threshold for r in retrieved)


def recall_for_question(
    retrieved: Sequence[MinimalSource],
    truths: Sequence[MinimalSource],
    k: int,
    min_overlap_ratio: float = 0.05,
) -> float:
    """Per-question recall: fraction of truth sources found in top-k."""
    if not truths:
        return 0.0
    topk = list(retrieved)[:k]
    found = sum(1 for t in truths if is_found(topk, t, min_overlap_ratio))
    return found / len(truths)


@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""

    questions_evaluated: int
    recalls: Dict[int, float]

    def render(self) -> str:
        lines = ["Evaluation Results", "=" * 40,
                 f"Questions evaluated: {self.questions_evaluated}"]
        for k in sorted(self.recalls):
            lines.append(f"Recall@{k}: {self.recalls[k]:.3f}")
        return "\n".join(lines)


def evaluate(
    student_results: StudentSearchResults,
    ground_truth: Iterable[AnsweredQuestion],
    ks: Sequence[int] = (1, 3, 5, 10),
    min_overlap_ratio: float = 0.05,
) -> EvaluationReport:
    """Compute recall@k for several values of ``k``."""
    by_id: Dict[str, MinimalSearchResults] = {
        r.question_id: r for r in student_results.search_results
    }
    totals: Dict[int, float] = {k: 0.0 for k in ks}
    count = 0
    for truth in ground_truth:
        if not truth.sources:
            continue
        pred = by_id.get(truth.question_id)
        if pred is None:
            continue
        count += 1
        for k in ks:
            totals[k] += recall_for_question(
                pred.retrieved_sources, truth.sources, k=k,
                min_overlap_ratio=min_overlap_ratio,
            )
    recalls: Dict[int, float] = {
        k: (totals[k] / count if count else 0.0) for k in ks
    }
    return EvaluationReport(questions_evaluated=count, recalls=recalls)


def load_ground_truth(path: str) -> List[AnsweredQuestion]:
    """Load an AnsweredQuestion dataset from disk."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions_raw = raw.get("rag_questions", raw)
    out: List[AnsweredQuestion] = []
    for q in questions_raw:
        if "sources" in q and "answer" in q:
            out.append(AnsweredQuestion.model_validate(q))
    return out
