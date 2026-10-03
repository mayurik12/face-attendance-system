from dataclasses import replace

import numpy as np
import pytest

from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from services.registration_service import RegistrationService
from tests.conftest import make_embedding, noisy_frame
from utils.errors import DuplicateUserError, RegistrationError, ValidationError

ONE_FACE = [(20, 80, 80, 20)]  # in downscaled coordinates (scale 0.5) -> 40px*... size 120px full-res


def make_service(db, settings, boxes=ONE_FACE, encoder=None):
    detector = FaceDetector(settings.frame_scale, backend=lambda rgb: list(boxes))
    encoder = encoder or FaceEncoder(backend=lambda rgb, b: [make_embedding(1, noise=0.01, noise_seed=np.random.randint(1e6))])
    return RegistrationService(db, detector, encoder, settings)


def test_successful_registration(db, settings):
    svc = make_service(db, settings)
    details = svc.validate_details("S100", " Asha  Rao ", "a@b.co", "CS")
    session = svc.new_session()
    for _ in range(settings.registration_samples):
        assert session.process_frame(noisy_frame()).accepted
    assert session.is_complete
    user = svc.complete(details, session)
    assert user.name == "Asha Rao"
    assert db.count_embeddings(user.id) == settings.registration_samples


def test_empty_name_and_bad_id_rejected(db, settings):
    svc = make_service(db, settings)
    with pytest.raises(ValidationError):
        svc.validate_details("S1x", "   ")
    with pytest.raises(ValidationError):
        svc.validate_details("", "Asha")


def test_duplicate_id_rejected(db, settings):
    svc = make_service(db, settings)
    db.add_user_with_embeddings("S100", "Existing", None, None, [make_embedding(3)])
    with pytest.raises(DuplicateUserError):
        svc.validate_details("S100", "Asha")


def test_insufficient_samples_does_not_register(db, settings):
    svc = make_service(db, settings)
    details = svc.validate_details("S100", "Asha")
    session = svc.new_session()
    session.process_frame(noisy_frame())
    with pytest.raises(RegistrationError):
        svc.complete(details, session)
    assert db.list_users() == []


def test_no_face_gives_useful_message(db, settings):
    session = make_service(db, settings, boxes=[]).new_session()
    result = session.process_frame(noisy_frame())
    assert not result.accepted and "No face" in result.message and not session.embeddings


def test_multiple_faces_asks_for_one(db, settings):
    session = make_service(db, settings, boxes=[(20, 80, 80, 20), (30, 90, 90, 40)]).new_session()
    result = session.process_frame(noisy_frame())
    assert not result.accepted and "only one face" in result.message


def test_blurry_sample_rejected(db, settings):
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = make_service(db, settings).new_session().process_frame(flat)
    assert not result.accepted and "blurry" in result.message


def test_small_face_rejected(db, settings):
    strict = replace(settings, min_face_size=100)
    session = make_service(db, strict, boxes=[(10, 40, 40, 10)]).new_session()  # 60px face at full res
    result = session.process_frame(noisy_frame())
    assert not result.accepted and "too small" in result.message


def test_invalid_image_does_not_crash(db, settings):
    session = make_service(db, settings).new_session()
    for bad in (None, "text", np.zeros((50, 50), dtype=np.uint8), np.zeros((0, 0, 3), dtype=np.uint8)):
        result = session.process_frame(bad)
        assert not result.accepted
