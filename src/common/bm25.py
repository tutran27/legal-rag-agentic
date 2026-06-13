import re
import zlib
from collections import Counter

from qdrant_client import models


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def bm25_vector(text: str, avg_length: float | None = None) -> models.SparseVector:
    tokens = tokenize(text)
    counts = Counter(tokens)
    length = len(tokens)
    weights = Counter()

    for token, frequency in counts.items():
        weight = float(frequency)
        if avg_length:
            weight = frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / avg_length)
            )
        weights[zlib.crc32(token.encode("utf-8"))] += weight

    pairs = sorted(weights.items())
    return models.SparseVector(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )


def average_length(texts) -> float:
    total = count = 0
    for text in texts:
        total += len(tokenize(text))
        count += 1
    return total / max(count, 1)
