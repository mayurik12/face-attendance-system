import numpy as np

from core.face_matcher import FaceMatcher, cosine_similarity
from tests.conftest import make_embedding


def loaded_matcher(threshold=0.5):
    m = FaceMatcher(threshold)
    m.load(np.array([1, 2]), np.vstack([make_embedding(1), make_embedding(2)]))
    return m


def test_identical_embedding_matches_with_zero_distance():
    r = loaded_matcher().match(make_embedding(1))
    assert r.is_known and r.user_id == 1 and r.distance == 0.0 and r.confidence == 1.0


def test_similar_embedding_matches_same_person():
    r = loaded_matcher().match(make_embedding(2, noise=0.01))
    assert r.is_known and r.user_id == 2 and 0 < r.distance < 0.5


def test_different_person_is_unknown():
    r = loaded_matcher().match(make_embedding(77))
    assert not r.is_known and r.user_id is None and r.distance > 0.5


def test_threshold_boundary_behaviour():
    query = make_embedding(1) + np.float32(0.05)
    d = float(np.linalg.norm(query - make_embedding(1)))
    m = loaded_matcher(threshold=d + 1e-4)
    assert m.match(query).is_known
    m.threshold = d - 1e-4
    assert not m.match(query).is_known


def test_stricter_threshold_rejects_borderline_match():
    query = make_embedding(1, noise=0.05)
    assert loaded_matcher(1.0).match(query).is_known
    assert not loaded_matcher(0.05).match(query).is_known


def test_empty_cache_returns_unknown():
    r = FaceMatcher(0.5).match(make_embedding(1))
    assert not r.is_known and r.distance is None


def test_nearest_of_several_samples_is_used():
    m = FaceMatcher(0.5)
    m.load(np.array([1, 1, 2]), np.vstack([make_embedding(1, 0.3, 5), make_embedding(1), make_embedding(2)]))
    assert m.match(make_embedding(1)).distance == 0.0


def test_confidence_decreases_with_distance():
    m = loaded_matcher()
    near = m.match(make_embedding(1, noise=0.005)).confidence
    far = m.match(make_embedding(1, noise=0.03)).confidence
    assert 1.0 > near > far >= 0.0


def test_cosine_similarity():
    a = make_embedding(1)
    assert cosine_similarity(a, a) > 0.999
    assert cosine_similarity(a, np.zeros_like(a)) == 0.0
