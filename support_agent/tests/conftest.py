import re

import numpy as np
import pytest

_DIM = 256


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    cjk = re.findall(r"[一-鿿]", lowered)
    return words + cjk


class FakeEncoder:
    """Deterministic bag-of-tokens encoder.

    Same text -> same vector; texts sharing tokens land close in cosine space,
    so retrieval tests are meaningful without downloading a real model.
    """

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.zeros((len(items), _DIM), dtype=np.float32)
        for row, text in enumerate(items):
            for token in _tokenize(text):
                bucket = hash(token) % _DIM
                vectors[row, bucket] += 1.0
        if single:
            return vectors[0]
        return vectors


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *args, **kwargs: FakeEncoder(),
    )
