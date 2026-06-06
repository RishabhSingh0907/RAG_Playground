"""
RAGAS evaluator backend.

Runtime evaluation using:

- Ollama
- OpenAI-compatible endpoint
- gpt-oss:20b
- RAGAS metrics
"""

from typing import List, Dict, Any, Optional
import math

from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from langchain_community.embeddings import OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# NEW metric imports (current API)
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)

from src.utils.logger import get_logger
from src.monitoring.ragas_adapter import build_ragas_inputs

logger = get_logger(__name__)


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None

        value = float(value)

        if math.isnan(value):
            return None

        return round(value, 4)

    except Exception:
        return None


class RagasEvaluator:
    """
    Runtime RAGAS evaluator.

    Uses:
    Ollama + OpenAI-compatible endpoint
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

        logger.info(
            "Initializing RagasEvaluator",
            model=model,
            base_url=base_url,
        )

        # OpenAI-compatible Ollama client
        self.client = OpenAI(
            api_key="ollama",
            base_url=f"{self.base_url}/v1",
        )

        # RAGAS LLM via OpenAI provider
        self.llm = llm_factory(
            model=model,
            provider="openai",
            client=self.client,
        )

        ollama_embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model="nomic-embed-text",
        )

        self.embeddings = LangchainEmbeddingsWrapper(
            ollama_embeddings
        )

        # Explicit metric objects
        self.faithfulness_metric = Faithfulness(
            llm=self.llm,
            # embeddings=self.embeddings,
        )

        self.relevancy_metric = ResponseRelevancy(
            llm=self.llm,
            embeddings=self.embeddings,
        )

        self.context_metric = (
            LLMContextPrecisionWithoutReference(
            llm=self.llm,
            # embeddings=self.embeddings,
            )
        )

        logger.info(
            "RAGAS metrics initialized",
            model=model,
        )

    def evaluate_response(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Any],
    ) -> Dict[str, Any]:
        """
        Evaluate generated response.

        Returns:
            Dict[str, Any]
        """

        empty_result = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "evaluator_model": self.model,
            "success": False,
        }

        try:
            payload = build_ragas_inputs(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
            )

            if (
                not payload["question"]
                or not payload["answer"]
                or not payload["contexts"]
            ):
                logger.warning(
                    "Skipping RAGAS evaluation due to empty inputs"
                )
                return empty_result

            dataset = Dataset.from_dict(
                {
                    "question": [
                        payload["question"]
                    ],
                    "answer": [
                        payload["answer"]
                    ],
                    "retrieved_contexts": [
                        payload["contexts"]
                    ],
                }
            )

            logger.info(
                "Running RAGAS evaluation",
                num_contexts=len(
                    payload["contexts"]
                ),
                model=self.model,
            )

            result = evaluate(
                dataset=dataset,
                metrics=[
                    self.faithfulness_metric,
                    self.relevancy_metric,
                    self.context_metric,
                ],
            )

            df = result.to_pandas()
            row = df.iloc[0]

            scores = {
                "faithfulness": _safe_float(
                    row.get("faithfulness")
                ),
                "answer_relevancy": _safe_float(
                    row.get(
                        "answer_relevancy"
                    )
                ),
                "context_precision": _safe_float(
                    row.get(
                        "llm_context_precision_without_reference"
                    )
                ),
                "evaluator_model": self.model,
                "success": True,
            }

            logger.info(
                "RAGAS evaluation complete",
                scores=scores,
            )

            return scores

        except Exception as exc:
            logger.error(
                "RAGAS evaluation failed",
                error=str(exc),
                exc_info=True,
            )

            empty_result["error"] = str(exc)
            return empty_result