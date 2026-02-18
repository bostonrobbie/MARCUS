"""
Tests for GlobalMemory class.

These tests verify the experience storage and retrieval system
using LLM embeddings for semantic search.
"""
import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestGlobalMemoryInitialization:
    """Tests for GlobalMemory initialization."""

    def test_creates_empty_memory(self, tmp_path, monkeypatch):
        """Should initialize with empty memory if file doesn't exist."""
        from memory import GlobalMemory

        # Point to non-existent file
        monkeypatch.setattr('memory.MEMORY_FILE', str(tmp_path / 'nonexistent.json'))

        with patch('memory.ollama'):
            mem = GlobalMemory()

        assert mem.memory == []

    def test_loads_existing_memory(self, tmp_path, monkeypatch):
        """Should load existing memory from file."""
        from memory import GlobalMemory

        # Create existing file
        memory_file = tmp_path / 'memory.json'
        existing = [{"summary": "Test", "outcome": "SUCCESS", "metrics": {}}]
        memory_file.write_text(json.dumps(existing))

        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama'):
            mem = GlobalMemory()

        assert len(mem.memory) == 1
        assert mem.memory[0]["summary"] == "Test"

    def test_handles_corrupted_file(self, tmp_path, monkeypatch):
        """Should return empty list on corrupted file."""
        from memory import GlobalMemory

        memory_file = tmp_path / 'memory.json'
        memory_file.write_text("not valid json{{{")

        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama'):
            mem = GlobalMemory()

        assert mem.memory == []


class TestAddExperience:
    """Tests for adding experiences to memory."""

    def test_adds_experience(self, tmp_path, monkeypatch):
        """Should add new experience with embedding."""
        from memory import GlobalMemory

        memory_file = tmp_path / 'memory.json'
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        mock_embedding = [0.1] * 1024

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": mock_embedding}
            mem = GlobalMemory()

            mem.add_experience(
                summary="Tested RSI period 14",
                outcome="SUCCESS",
                metrics={"sharpe": 1.5}
            )

        assert len(mem.memory) == 1
        assert mem.memory[0]["summary"] == "Tested RSI period 14"
        assert mem.memory[0]["outcome"] == "SUCCESS"
        assert "timestamp" in mem.memory[0]
        assert "embedding" in mem.memory[0]

    def test_saves_to_file(self, tmp_path, monkeypatch):
        """Should persist experience to file."""
        from memory import GlobalMemory

        memory_file = tmp_path / 'memory.json'
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.1] * 1024}
            mem = GlobalMemory()
            mem.add_experience("Test", "SUCCESS", {})

        # Read file directly
        saved = json.loads(memory_file.read_text())
        assert len(saved) == 1


class TestQueryMemory:
    """Tests for querying memory."""

    def test_returns_empty_for_empty_memory(self, tmp_path, monkeypatch):
        """Should return empty list when no experiences."""
        from memory import GlobalMemory

        monkeypatch.setattr('memory.MEMORY_FILE', str(tmp_path / 'memory.json'))

        with patch('memory.ollama'):
            mem = GlobalMemory()
            results = mem.query_memory("test query")

        assert results == []

    def test_returns_similar_experiences(self, tmp_path, monkeypatch):
        """Should return experiences above similarity threshold."""
        from memory import GlobalMemory

        # Create memory with known embeddings
        memory_file = tmp_path / 'memory.json'
        existing = [
            {"summary": "RSI strategy test", "outcome": "SUCCESS", "embedding": [1.0] + [0.0] * 1023},
            {"summary": "MACD strategy test", "outcome": "FAILURE", "embedding": [0.5, 0.5] + [0.0] * 1022}
        ]
        memory_file.write_text(json.dumps(existing))
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama') as mock_ollama:
            # Query embedding similar to first entry
            mock_ollama.embeddings.return_value = {"embedding": [0.9] + [0.1] * 1023}
            mem = GlobalMemory()
            results = mem.query_memory("RSI test", threshold=0.5)

        assert len(results) >= 1

    def test_returns_top_3_results(self, tmp_path, monkeypatch):
        """Should return at most 3 results."""
        from memory import GlobalMemory

        # Create many similar entries
        memory_file = tmp_path / 'memory.json'
        existing = [
            {"summary": f"Test {i}", "outcome": "SUCCESS", "embedding": [0.9] * 1024}
            for i in range(10)
        ]
        memory_file.write_text(json.dumps(existing))
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.9] * 1024}
            mem = GlobalMemory()
            results = mem.query_memory("test", threshold=0.5)

        assert len(results) <= 3

    def test_skips_entries_without_embedding(self, tmp_path, monkeypatch):
        """Should skip entries that lack embeddings."""
        from memory import GlobalMemory

        memory_file = tmp_path / 'memory.json'
        existing = [
            {"summary": "No embedding", "outcome": "SUCCESS"},  # No embedding
            {"summary": "Has embedding", "outcome": "SUCCESS", "embedding": [0.9] * 1024}
        ]
        memory_file.write_text(json.dumps(existing))
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.9] * 1024}
            mem = GlobalMemory()
            results = mem.query_memory("test", threshold=0.5)

        # Should not crash, just skip the one without embedding
        assert len(results) <= 2


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    def test_calls_ollama(self, tmp_path, monkeypatch):
        """Should call Ollama for embeddings."""
        from memory import GlobalMemory

        monkeypatch.setattr('memory.MEMORY_FILE', str(tmp_path / 'memory.json'))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.5] * 1024}
            mem = GlobalMemory(model="test-model")
            mem._get_embedding("test text")

        mock_ollama.embeddings.assert_called_with(model="test-model", prompt="test text")

    def test_returns_fallback_on_error(self, tmp_path, monkeypatch):
        """Should return zero vector on embedding error."""
        from memory import GlobalMemory

        monkeypatch.setattr('memory.MEMORY_FILE', str(tmp_path / 'memory.json'))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.side_effect = Exception("API error")
            mem = GlobalMemory()
            result = mem._get_embedding("test")

        assert result == [0.0] * 1024
