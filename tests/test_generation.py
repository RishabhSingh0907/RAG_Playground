"""Tests for generation layer components."""

import pytest
from datetime import datetime

from src.generation.llm_provider import LLMConfig, LLMProvider, OllamaProvider
from src.generation.prompt_builder import PromptBuilder, format_retrieved_chunks
from src.generation.citation_extractor import CitationExtractor, build_chunk_dict_from_retrieved, Citation
from src.generation.chat_manager import ChatManager, ChatMessage, ChatSession
from src.retrieval.pipeline import RetrievedChunk


# ==================== Fixtures ====================

@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks for testing."""
    return [
        RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            score=0.95,
            rank=1,
            source="dense",
            metadata={"file_name": "ai_guide.pdf", "file_path": "/docs/ai_guide.pdf"},
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="Neural networks are inspired by biological neurons and can process complex patterns.",
            score=0.87,
            rank=2,
            source="dense",
            metadata={"file_name": "neural_networks.pdf", "file_path": "/docs/neural_networks.pdf"},
        ),
    ]


@pytest.fixture
def llm_config():
    """Create test LLM config."""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434",
        model="mistral",
        temperature=0.7,
        max_tokens=1024,
    )


# ==================== Tests: PromptBuilder ====================

class TestPromptBuilder:
    """Test prompt building functionality."""

    def test_format_retrieved_chunks(self, sample_chunks):
        """Test formatting chunks into context."""
        context = format_retrieved_chunks(sample_chunks)
        
        assert "chunk_1" in context
        assert "chunk_2" in context
        assert "Machine learning" in context
        assert "Neural networks" in context
        assert "[SOURCE:" in context

    def test_build_prompt(self, sample_chunks):
        """Test building a complete prompt."""
        builder = PromptBuilder()
        query = "What is machine learning?"
        
        prompt = builder.build_prompt(query, sample_chunks)
        
        assert "system prompt" not in prompt.lower() or "assistant" in prompt.lower()
        assert query in prompt
        assert "chunk_1" in prompt
        assert "chunk_2" in prompt

    def test_minimal_prompt(self, sample_chunks):
        """Test building minimal prompt."""
        builder = PromptBuilder()
        query = "What is machine learning?"
        
        prompt = builder.build_minimal_prompt(query, sample_chunks)
        
        assert query in prompt
        assert len(prompt) < 1000  # Should be shorter than full prompt

    def test_prompt_stats(self, sample_chunks):
        """Test prompt statistics calculation."""
        builder = PromptBuilder()
        query = "What is machine learning?"
        prompt = builder.build_prompt(query, sample_chunks)
        
        stats = builder.get_prompt_stats(prompt)
        
        assert "character_count" in stats
        assert "estimated_tokens" in stats
        assert stats["character_count"] == len(prompt)
        assert stats["estimated_tokens"] > 0


# ==================== Tests: CitationExtractor ====================

class TestCitationExtractor:
    """Test citation extraction functionality."""

    def test_extract_citations(self):
        """Test extracting citations from response."""
        extractor = CitationExtractor()
        response = "Machine learning is a subset of AI [CITE: ai_guide.pdf | chunk_1] and neural networks are important [CITE: neural_networks.pdf | chunk_2]"
        
        citations = extractor.extract_citations(response)
        
        assert len(citations) == 2
        assert citations[0].doc_name == "ai_guide.pdf"
        assert citations[0].chunk_id == "1"
        assert citations[1].doc_name == "neural_networks.pdf"
        assert citations[1].chunk_id == "2"

    def test_extract_source_pattern(self):
        """Test extracting SOURCE pattern citations."""
        extractor = CitationExtractor()
        response = "ML is AI [SOURCE: ai_guide.pdf | chunk_1]"
        
        citations = extractor.extract_citations(response)
        
        assert len(citations) == 1
        assert citations[0].doc_name == "ai_guide.pdf"
        assert citations[0].chunk_id == "1"

    def test_clean_response(self):
        """Test removing citation tags from response."""
        extractor = CitationExtractor()
        response = "Machine learning [CITE: doc.pdf | chunk_1] is important"
        
        cleaned = extractor.clean_response(response)
        
        assert "[CITE:" not in cleaned
        assert "Machine learning" in cleaned
        assert "is important" in cleaned

    def test_link_citations_to_chunks(self, sample_chunks):
        """Test linking citations to chunk data."""
        extractor = CitationExtractor()
        chunk_dict = build_chunk_dict_from_retrieved(sample_chunks)
        
        citations = [
            Citation(doc_name="ai_guide.pdf", chunk_id="chunk_1", position=0, citation_text="[CITE: ai_guide.pdf | chunk_1]"),
        ]
        
        linked = extractor.link_citations_to_chunks(citations, chunk_dict)
        
        assert len(linked) == 1
        assert linked[0].chunk_content == sample_chunks[0].content
        assert linked[0].chunk_metadata["file_name"] == "ai_guide.pdf"

    def test_format_response_with_tags(self):
        """Test formatting response with citation tags."""
        extractor = CitationExtractor()
        response = "ML is AI [CITE: doc.pdf | chunk_1]"
        citations = [
            Citation(doc_name="doc.pdf", chunk_id="1", position=8, citation_text="[CITE: doc.pdf | chunk_1]"),
        ]
        
        formatted = extractor.format_response_with_tags(response, citations)
        
        assert "<cite" in formatted
        assert 'data-doc="doc.pdf"' in formatted
        assert 'data-chunk="1"' in formatted


# ==================== Tests: ChatManager ====================

class TestChatManager:
    """Test chat management functionality."""

    def test_initialize_chat_manager(self):
        """Test chat manager initialization."""
        manager = ChatManager(session_id="test_session")
        
        assert manager.current_session.session_id == "test_session"
        assert len(manager.current_session.messages) == 0

    def test_add_user_message(self):
        """Test adding user message."""
        manager = ChatManager()
        msg = manager.add_user_message("What is ML?")
        
        assert msg.role == "user"
        assert msg.content == "What is ML?"
        assert len(manager.current_session.messages) == 1

    def test_add_assistant_message(self, sample_chunks):
        """Test adding assistant message with context."""
        manager = ChatManager()
        manager.add_user_message("What is ML?")
        
        citations = [Citation(doc_name="ai_guide.pdf", chunk_id="chunk_1", position=0, citation_text="[CITE: ai_guide.pdf | chunk_1]")]
        msg = manager.add_assistant_message(
            "Machine learning is AI [CITE: ai_guide.pdf | chunk_1]",
            retrieved_chunks=sample_chunks,
            citations=citations,
        )
        
        assert msg.role == "assistant"
        assert len(msg.retrieved_chunks) == 2
        assert len(msg.citations) == 1

    def test_conversation_history(self):
        """Test retrieving conversation history."""
        manager = ChatManager()
        manager.add_user_message("Question 1?")
        manager.add_assistant_message("Answer 1")
        
        history = manager.get_conversation_history()
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_session_summary(self, sample_chunks):
        """Test getting session summary."""
        manager = ChatManager()
        manager.add_user_message("What is ML?")
        
        citations = [Citation(doc_name="ai_guide.pdf", chunk_id="chunk_1", position=0, citation_text="[CITE: ai_guide.pdf | chunk_1]")]
        manager.add_assistant_message(
            "ML is AI [CITE: ai_guide.pdf | chunk_1]",
            retrieved_chunks=sample_chunks,
            citations=citations,
        )
        
        summary = manager.get_session_summary()
        
        assert summary["num_turns"] == 1
        assert summary["num_user_messages"] == 1
        assert summary["num_assistant_messages"] == 1
        assert summary["total_chunks_retrieved"] == 2
        assert summary["total_citations"] == 1

    def test_clear_session(self):
        """Test clearing session."""
        manager = ChatManager()
        manager.add_user_message("Question?")
        manager.add_assistant_message("Answer")
        
        assert len(manager.current_session.messages) == 2
        
        manager.clear_session()
        
        assert len(manager.current_session.messages) == 0

    def test_get_context_for_last_query(self, sample_chunks):
        """Test retrieving context for last query."""
        manager = ChatManager()
        manager.add_user_message("What is ML?")
        manager.add_assistant_message("ML answer", retrieved_chunks=sample_chunks)
        
        context = manager.get_context_for_last_query()
        
        assert context is not None
        assert len(context) == 2


# ==================== Tests: LLMConfig ====================

class TestLLMConfig:
    """Test LLM configuration."""

    def test_llm_config_initialization(self, llm_config):
        """Test LLM config initialization."""
        assert llm_config.provider == LLMProvider.OLLAMA
        assert llm_config.model == "mistral"
        assert llm_config.temperature == 0.7

    def test_llm_config_to_dict(self, llm_config):
        """Test converting config to dict."""
        config_dict = llm_config.to_dict()
        
        assert config_dict["provider"] == "ollama"
        assert config_dict["model"] == "mistral"
        assert config_dict["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
