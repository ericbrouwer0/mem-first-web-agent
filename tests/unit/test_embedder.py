"""Unit tests for the Embedder."""

from unittest.mock import MagicMock, patch

from src.memory.embedder import Embedder


def _fake_embedding(dim=1536):
    return [0.01] * dim


def test_embed_text_returns_correct_dimension():
    with patch("src.memory.embedder.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        fake_resp = MagicMock()
        fake_data = MagicMock()
        fake_data.embedding = _fake_embedding()
        fake_resp.data = [fake_data]
        mock_client.embeddings.create.return_value = fake_resp

        embedder = Embedder()
        result = embedder.embed_text("hello world")
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)


def test_embed_batch_preserves_order():
    with patch("src.memory.embedder.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        items = []
        for i in range(3):
            d = MagicMock()
            d.index = i
            d.embedding = _fake_embedding()
            items.append(d)

        fake_resp = MagicMock()
        fake_resp.data = items
        mock_client.embeddings.create.return_value = fake_resp

        embedder = Embedder()
        result = embedder.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 1536


def test_embed_batch_empty():
    embedder = Embedder.__new__(Embedder)
    result = embedder.embed_batch([])
    assert result == []
