"""Monitoring and evaluation module for the RAG Playground.

Exports:
    EvaluationResult  – data schema for a single evaluation run
    RAGASEvaluator    – runs RAGAS metrics via a local Ollama model
    SessionMonitor    – in-session evaluation history store
"""

# from src.monitoring.schemas import EvaluationResult
# from src.monitoring.evaluator import RAGASEvaluator
# from src.monitoring.session_monitor import SessionMonitor

# __all__ = ["EvaluationResult", "RAGASEvaluator", "SessionMonitor"]

# from .evaluator import evaluate_response

# __all__ = ["evaluate_response"]

"""
Monitoring package for RAGAS evaluation.
"""

from .evaluator import RagasEvaluator
from .ragas_adapter import build_ragas_inputs

__all__ = [
    "RagasEvaluator",
    "build_ragas_inputs",
]