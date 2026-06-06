"""Tests for the local RAGAS monitoring layer."""

from datetime import datetime

from src.monitoring.evaluator import MonitoringConfig, MonitoringEvaluator, load_monitoring_config
from src.monitoring.schemas import EvaluationResult
from src.monitoring.session_monitor import SessionMonitor


def test_evaluation_result_roundtrip():
    """EvaluationResult should serialize and deserialize cleanly."""

    result = EvaluationResult(
        question="What is RAG?",
        answer="Retrieval augmented generation.",
        contexts=["RAG combines retrieval and generation."],
        faithfulness=0.91,
        answer_relevancy=0.88,
        context_precision=0.84,
        evaluator_model="gpt-oss:20b",
        evaluator_base_url="http://localhost:11434",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        success=True,
    )

    restored = EvaluationResult.from_dict(result.to_dict())

    assert restored.question == result.question
    assert restored.answer == result.answer
    assert restored.contexts == result.contexts
    assert restored.faithfulness == result.faithfulness
    assert restored.answer_relevancy == result.answer_relevancy
    assert restored.context_precision == result.context_precision
    assert restored.evaluator_model == result.evaluator_model
    assert restored.success is True


def test_monitoring_config_loader_uses_yaml_mapping():
    """load_monitoring_config should map the YAML monitoring section correctly."""

    config = load_monitoring_config("does-not-exist.yaml")

    assert config.enabled is True
    assert config.evaluator_model == "gpt-oss:20b"
    assert config.embedding_model == "nomic-embed-text"


def test_evaluate_response_returns_structured_scores(monkeypatch):
    """The evaluator should package metric scores into a structured result."""

    def fake_run(self, question, answer, contexts):
        return (
            {
                "faithfulness": 0.93,
                "answer_relevancy": 0.89,
                "context_precision": 0.86,
            },
            {
                "faithfulness": 0.93,
                "answer_relevancy": 0.89,
                "context_precision": 0.86,
            },
        )

    monkeypatch.setattr(MonitoringEvaluator, "_run_ragas_evaluation", fake_run)

    evaluator = MonitoringEvaluator(MonitoringConfig(enabled=True))
    result = evaluator.evaluate_response(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        contexts=["RAG uses retrieved documents to ground answers."],
    )

    assert result.success is True
    assert result.faithfulness == 0.93
    assert result.answer_relevancy == 0.89
    assert result.context_precision == 0.86
    assert result.error_message is None


def test_evaluate_response_handles_missing_contexts():
    """The evaluator should fail gracefully when contexts are absent."""

    evaluator = MonitoringEvaluator(MonitoringConfig(enabled=True))

    result = evaluator.evaluate_response(
        question="What is RAG?",
        answer="Retrieval augmented generation.",
        contexts=[],
    )

    assert result.success is False
    assert "No retrieved contexts" in result.error_message


def test_session_monitor_keeps_recent_history_and_averages():
    """SessionMonitor should retain a bounded history and compute averages."""

    monitor = SessionMonitor(max_history=2)

    first = EvaluationResult(
        question="Q1",
        answer="A1",
        contexts=["C1"],
        faithfulness=0.5,
        answer_relevancy=0.6,
        context_precision=0.7,
        success=True,
    )
    second = EvaluationResult(
        question="Q2",
        answer="A2",
        contexts=["C2"],
        faithfulness=0.8,
        answer_relevancy=0.9,
        context_precision=1.0,
        success=True,
    )
    third = EvaluationResult(
        question="Q3",
        answer="A3",
        contexts=["C3"],
        faithfulness=0.9,
        answer_relevancy=0.95,
        context_precision=0.85,
        success=True,
    )

    monitor.add_result(first)
    monitor.add_result(second)
    monitor.add_result(third)

    assert len(monitor.evaluation_history) == 2
    assert monitor.latest_result().question == "Q3"
    assert [item.question for item in monitor.recent_results(limit=2)] == ["Q3", "Q2"]

    averages = monitor.average_scores()
    assert averages["faithfulness"] == 0.85
    assert averages["answer_relevancy"] == 0.925
    assert averages["context_precision"] == 0.925
