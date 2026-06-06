"""Data schemas for RAGAS evaluation results.

Defines clean contracts between the evaluator, session monitor, and UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class EvaluationResult:
    """Structured result from a single RAGAS evaluation run.

    All metric scores are in the range [0, 1] where higher is better.
    A score of None indicates the metric could not be computed (e.g. LLM error).
    """

    # Core RAGAS metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None

    # Metadata
    evaluator_model: str = "gpt-oss:20b"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Evaluation inputs (stored for session history display)
    question: str = ""
    answer: str = ""
    num_contexts: int = 0

    # Error state
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session storage or export."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "evaluator_model": self.evaluator_model,
            "timestamp": self.timestamp.isoformat(),
            "question": self.question[:120] + ("..." if len(self.question) > 120 else ""),
            "answer_preview": self.answer[:120] + ("..." if len(self.answer) > 120 else ""),
            "num_contexts": self.num_contexts,
            "success": self.success,
            "error_message": self.error_message,
        }

    def overall_score(self) -> Optional[float]:
        """Compute simple average of available scores."""
        scores = [
            s for s in [self.faithfulness, self.answer_relevancy, self.context_precision]
            if s is not None
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 3)

    @staticmethod
    def failed(error_message: str, question: str = "", evaluator_model: str = "gpt-oss:20b") -> "EvaluationResult":
        """Construct a failure result."""
        return EvaluationResult(
            success=False,
            error_message=error_message,
            question=question,
            evaluator_model=evaluator_model,
        )
