"""Chat manager for handling conversation state and message tracking.

Responsibilities:
- Store chat messages (user + assistant)
- Track retrieved context per message
- Manage session-based conversations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.retrieval.pipeline import RetrievedChunk
from src.generation.citation_extractor import Citation
from src.utils.logger import get_logger

from src.generation.llm_provider import (
    LLMConfig,
    create_llm_provider,
)

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    citations: Optional[List[Citation]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI/persistence."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "num_retrieved_chunks": len(self.retrieved_chunks) if self.retrieved_chunks else 0,
            "num_citations": len(self.citations) if self.citations else 0,
            "metadata": self.metadata,
        }


class ChatSession:
    """Manages a single chat session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

    def add_message(
        self,
        role: str,
        content: str,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        citations: Optional[List[Citation]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add a message to the session.

        Args:
            role: "user" or "assistant"
            content: Message content
            retrieved_chunks: Retrieved chunks (for assistant messages)
            citations: Extracted citations (for assistant messages)
            metadata: Additional metadata

        Returns:
            Created ChatMessage
        """
        message = ChatMessage(
            role=role,
            content=content,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.last_activity = datetime.utcnow()

        logger.info(
            "Message added to session",
            session_id=self.session_id,
            role=role,
            content_length=len(content),
        )

        return message

    def get_last_user_message(self) -> Optional[ChatMessage]:
        """Get the last user message."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None

    def get_last_assistant_message(self) -> Optional[ChatMessage]:
        """Get the last assistant message."""
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        return None

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation."""
        user_messages = [m for m in self.messages if m.role == "user"]
        assistant_messages = [m for m in self.messages if m.role == "assistant"]
        total_chunks_retrieved = sum(
            len(m.retrieved_chunks) if m.retrieved_chunks else 0
            for m in assistant_messages
        )
        total_citations = sum(
            len(m.citations) if m.citations else 0 for m in assistant_messages
        )

        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "num_turns": len(user_messages),
            "num_user_messages": len(user_messages),
            "num_assistant_messages": len(assistant_messages),
            "total_chunks_retrieved": total_chunks_retrieved,
            "total_citations": total_citations,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "num_messages": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
        }


class ChatManager:
    """Manage chat sessions (single session for Phase 1)."""

    def __init__(self, session_id: str = "default"):
        self.current_session = ChatSession(session_id)
        self.sessions: Dict[str, ChatSession] = {session_id: self.current_session}

        logger.info("ChatManager initialized", session_id=session_id)

    def add_user_message(self, content: str) -> ChatMessage:
        """Add a user message to current session.

        Args:
            content: User message

        Returns:
            Created ChatMessage
        """
        return self.current_session.add_message(role="user", content=content)

    def add_assistant_message(
        self,
        content: str,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        citations: Optional[List[Citation]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add an assistant message to current session.

        Args:
            content: Assistant response
            retrieved_chunks: Retrieved context
            citations: Extracted citations
            metadata: Additional metadata

        Returns:
            Created ChatMessage
        """
        return self.current_session.add_message(
            role="assistant",
            content=content,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            metadata=metadata,
        )

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history from current session.

        Returns:
            List of message dicts
        """
        return [m.to_dict() for m in self.current_session.messages]

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return self.current_session.get_conversation_summary()

    def clear_session(self) -> None:
        """Clear current session (for starting new conversation)."""
        session_id = self.current_session.session_id
        self.current_session = ChatSession(session_id)
        self.sessions[session_id] = self.current_session

        logger.info("Session cleared", session_id=session_id)

    def get_available_models(
        self,
        base_url: str = "http://localhost:11434",
        ):
        """
        Get available Ollama models.

        Args:
            base_url: Ollama server URL

        Returns:
            List of model names
        """
        try:
            config = LLMConfig(
                base_url=base_url
            )

            provider = create_llm_provider(config)

            if provider:
                return provider.get_available_models()

            return []

        except Exception as exc:
            logger.error(
                "Failed to fetch available models",
                error=str(exc),
                exc_info=True,
            )
            return []

    def get_context_for_last_query(self) -> Optional[List[RetrievedChunk]]:
        """Get retrieved context for the last user query.

        Returns:
            List of chunks retrieved for last query, or None
        """
        last_assistant_message = self.current_session.get_last_assistant_message()
        if last_assistant_message and last_assistant_message.retrieved_chunks:
            return last_assistant_message.retrieved_chunks
        return None
