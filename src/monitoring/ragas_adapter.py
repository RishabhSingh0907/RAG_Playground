"""
RAGAS adapter.

Converts playground retrieval outputs
into RAGAS-compatible inputs.
"""

from typing import List, Dict, Any

from src.retrieval.pipeline import RetrievedChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_ragas_inputs(
    question: str,
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """
    Convert retrieved chunks to RAGAS inputs.

    Args:
        question:
            User query

        answer:
            Generated response

        retrieved_chunks:
            RetrievedChunk objects

    Returns:
        Dict for evaluator
    """

    contexts = []

    for chunk in retrieved_chunks or []:
        try:
            contexts.append(chunk.content)
        except Exception:
            continue

    logger.info(
        "RAGAS inputs prepared",
        num_contexts=len(contexts),
        question_length=len(question),
        answer_length=len(answer),
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }