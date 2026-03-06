"""Tests for the Embedding analytical layer."""

import torch
import pytest
from to_the_point import Embedding


class TestEmbeddingForward:
    def test_output_shape(self):
        layer = Embedding(vocab_size=100, embed_dim=16)
        x = torch.randint(0, 100, (4, 10))
        out = layer.forward(x)
        assert out.shape == (4, 10, 16)

    def test_positional_embedding(self):
        layer = Embedding(
            vocab_size=100, embed_dim=16, use_positional=True
        )
        x = torch.randint(0, 100, (4, 10))
        out = layer.forward(x)
        assert out.shape == (4, 10, 16)

    def test_get_positional_embeddings(self):
        layer = Embedding(
            vocab_size=100,
            embed_dim=16,
            max_seq_len=64,
            use_positional=True,
        )
        pe = layer.get_positional_embeddings(seq_len=10)
        assert pe.shape == (10, 16)

    def test_no_positional_returns_none(self):
        layer = Embedding(vocab_size=100, embed_dim=16, use_positional=False)
        assert layer.get_positional_embeddings() is None
