"""
Medical RAG Chain
LangChain LCEL-based chain with strict anti-hallucination enforcement.
Temperature 0.0 for deterministic, grounded outputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.generation.prompts import MEDICAL_RAG_PROMPT, INSUFFICIENT_DATA_RESPONSE
from app.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "Low"
    max_score: float = 0.0
    retrieval_strategy: str = "mmr"
    low_confidence_fallback: bool = False


def get_llm():
    """Initialize Mistral LLM with deterministic settings."""
    if not settings.MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. "
            "Please add it to your .env file or set USE_LOCAL_EMBEDDINGS=True "
            "and configure a local LLM."
        )
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model=settings.MISTRAL_LLM_MODEL,
        mistral_api_key=settings.MISTRAL_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        random_seed=42,  # deterministic outputs
    )


class MedicalRAGChain:
    """
    The core RAG generation chain.
    Takes a RetrievalResult and generates a grounded, cited response.
    """

    def __init__(self):
        self._llm = get_llm()
        self._chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | MEDICAL_RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )
        logger.info(f"RAGChain initialized with model: {settings.MISTRAL_LLM_MODEL}")

    def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> RAGResponse:
        """
        Generate a grounded medical answer from retrieved context.
        
        If retrieval confidence is low, returns the insufficient data fallback
        immediately without calling the LLM (saves API cost + prevents hallucination).
        """
        # ── Confidence Gate ───────────────────────────────────────────────
        if not retrieval_result.is_confident or not retrieval_result.documents:
            logger.warning(
                f"Low confidence retrieval (score={retrieval_result.max_score:.3f}). "
                f"Returning fallback response."
            )
            return RAGResponse(
                answer=INSUFFICIENT_DATA_RESPONSE,
                sources=[],
                confidence="Low",
                max_score=retrieval_result.max_score,
                retrieval_strategy=retrieval_result.strategy,
                low_confidence_fallback=True,
            )

        # ── LLM Generation ────────────────────────────────────────────────
        context = retrieval_result.context_text
        logger.info(
            f"Generating answer | docs={len(retrieval_result.documents)} | "
            f"context_len={len(context)} chars"
        )

        try:
            raw_answer = self._chain.invoke({
                "context": context,
                "question": question,
            })
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            raise

        # ── Determine confidence from score ───────────────────────────────
        score = retrieval_result.max_score
        if score >= 0.75:
            confidence = "High"
        elif score >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"

        return RAGResponse(
            answer=raw_answer,
            sources=retrieval_result.source_citations,
            confidence=confidence,
            max_score=score,
            retrieval_strategy=retrieval_result.strategy,
            low_confidence_fallback=False,
        )
