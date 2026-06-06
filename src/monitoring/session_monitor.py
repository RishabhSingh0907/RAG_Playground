"""Session-level evaluation history and aggregate stats.

Responsibilities:
- Store EvaluationResult objects for the active Streamlit session
- Expose the last result and session-level averages for the UI
- Remain stateless beyond the session (no DB writes)

Usage:
    monitor = SessionMonitor()
    monitor.record(result)

    last   = monitor.last_evaluation()
    avgs   = monitor.average_scores()
    recent = monitor.recent_evaluations(n=5)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from src.monitoring.schemas import EvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionMonitor:
    """Lightweight in-session store for evaluation results.

    Designed to live in st.session_state so it survives Streamlit reruns
    but resets on a full page reload / new session.
    """

    def __init__(self) -> None:
        self._history: List[EvaluationResult] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, result: EvaluationResult) -> None:
        """Append an evaluation result to the history.

        Args:
            result: The EvaluationResult produced after a generation turn.
        """
        self._history.append(result)
        logger.info(
            "Evaluation result recorded",
            total_evals=len(self._history),
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            success=result.success,
        )

    def clear(self) -> None:
        """Wipe evaluation history (e.g. when user clears the chat)."""
        self._history.clear()
        logger.info("Evaluation history cleared")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def last_evaluation(self) -> Optional[EvaluationResult]:
        """Return the most recent evaluation result, or None."""
        if not self._history:
            return None
        return self._history[-1]

    def recent_evaluations(self, n: int = 5) -> List[EvaluationResult]:
        """Return the last *n* evaluation results, newest first.

        Args:
            n: How many results to return.

        Returns:
            List of EvaluationResult objects, most recent first.
        """
        return list(reversed(self._history[-n:]))

    def total_evaluations(self) -> int:
        """Total number of evaluations recorded this session."""
        return len(self._history)

    def average_scores(self) -> Dict[str, Optional[float]]:
        """Compute session-level averages across successful evaluations.

        Returns:
            Dict with keys: faithfulness, answer_relevancy, context_precision.
            Each value is a float average or None if no successful evals exist.
        """
        successful = [r for r in self._history if r.success]

        if not successful:
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            }

        def _avg(attr: str) -> Optional[float]:
            vals = [getattr(r, attr) for r in successful if getattr(r, attr) is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 3)

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
        }

    def session_summary(self) -> Dict[str, Any]:
        """Return a compact summary dict for sidebar/status display."""
        avgs = self.average_scores()
        last = self.last_evaluation()

        return {
            "total_evaluations": self.total_evaluations(),
            "successful_evaluations": len([r for r in self._history if r.success]),
            "average_faithfulness": avgs["faithfulness"],
            "average_answer_relevancy": avgs["answer_relevancy"],
            "average_context_precision": avgs["context_precision"],
            "last_overall_score": last.overall_score() if last else None,
        }
