"""Compare a query embedding with cached known embeddings.

Distance: Euclidean (L2). dlib embeddings are trained so that two photos of the
same person are close (typically < ~0.6) and different people are further away.
Decision rule: nearest stored sample; match only if distance <= threshold.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from config.settings import EMBEDDING_DIM


@dataclass(frozen=True)
class MatchResult:
    user_id: int | None
    distance: float | None
    is_known: bool
    confidence: float  # 0..1 heuristic derived from distance; NOT a probability


def euclidean_distances(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix - query, axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Shown for comparison/explanation; the matcher itself uses Euclidean distance."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class FaceMatcher:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._lock = threading.Lock()

    def load(self, user_ids: np.ndarray, embeddings: np.ndarray) -> None:
        """Replace the in-memory cache (called once, not per frame)."""
        ids = np.asarray(user_ids, dtype=np.int64)
        matrix = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        if len(ids) != len(matrix):
            raise ValueError("user_ids and embeddings must have the same length.")
        with self._lock:
            self._ids, self._matrix = ids, matrix

    @property
    def sample_count(self) -> int:
        return len(self._ids)

    def match(self, embedding: np.ndarray) -> MatchResult:
        query = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            ids, matrix = self._ids, self._matrix
        if len(ids) == 0:
            return MatchResult(None, None, False, 0.0)
        distances = euclidean_distances(matrix, query)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        # d=0 -> 1.0, d=threshold -> 0.5, d>=2*threshold -> 0.0 (a display heuristic only)
        confidence = float(np.clip(1.0 - distance / (2.0 * self.threshold), 0.0, 1.0))
        if distance <= self.threshold:
            return MatchResult(int(ids[best]), distance, True, confidence)
        return MatchResult(None, distance, False, confidence)
