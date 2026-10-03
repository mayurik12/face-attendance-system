import numpy as np
import pytest

from config.settings import Settings
from core.attendance import AttendanceManager
from core.camera import Camera
from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.face_matcher import FaceMatcher
from services.export_service import export_attendance_csv
from services.recognition_service import RecognitionService
from tests.conftest import make_embedding, noisy_frame
from utils.errors import (CameraError, ConfigError, InvalidImageError, ModelUnavailableError,
                          MultipleFacesError, NoFaceError)


class FakeCapture:
    def __init__(self, opened=True, frames=1):
        self._opened, self._frames, self.released = opened, frames, False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, noisy_frame()

    def release(self):
        self.released = True


# ---- camera -----------------------------------------------------------------
def test_camera_unavailable_raises_friendly_error():
    cap = FakeCapture(opened=False)
    with pytest.raises(CameraError, match="not available"):
        Camera(0, lambda i: cap).open()
    assert cap.released


def test_camera_factory_exception_wrapped():
    def boom(_):
        raise RuntimeError("driver crash")
    with pytest.raises(CameraError):
        Camera(0, boom).open()


def test_camera_lifecycle_and_lost_frame():
    cap = FakeCapture(frames=1)
    cam = Camera(0, lambda i: cap)
    with pytest.raises(CameraError):
        cam.read()  # not open yet
    with cam:
        assert cam.read().shape == (200, 200, 3)
        with pytest.raises(CameraError):
            cam.read()  # camera stopped delivering frames
    assert cap.released and not cam.is_open


# ---- detector / encoder -----------------------------------------------------
def test_detect_single_errors():
    frame = noisy_frame()
    with pytest.raises(NoFaceError):
        FaceDetector(0.5, backend=lambda rgb: []).detect_single(frame)
    with pytest.raises(MultipleFacesError):
        FaceDetector(0.5, backend=lambda rgb: [(1, 5, 5, 1), (2, 6, 6, 2)]).detect_single(frame)


def test_boxes_scaled_back_to_full_resolution():
    box = FaceDetector(0.5, backend=lambda rgb: [(10, 40, 40, 10)]).detect(noisy_frame())[0]
    assert box == (20, 80, 80, 20)


def test_detector_gets_downscaled_rgb_image():
    seen = {}
    FaceDetector(0.25, backend=lambda rgb: seen.setdefault("shape", rgb.shape) and []).detect(noisy_frame(200, 400))
    assert seen["shape"] == (50, 100, 3)


@pytest.mark.parametrize("bad", [None, [1, 2], np.zeros((100, 100)), np.zeros((100, 100, 3), dtype=np.float32), np.zeros((2, 2, 3), dtype=np.uint8)])
def test_invalid_images_rejected(bad):
    with pytest.raises(InvalidImageError):
        FaceDetector(0.5, backend=lambda rgb: []).detect(bad)


def test_encoder_failures():
    with pytest.raises(InvalidImageError):
        FaceEncoder(backend=lambda rgb, b: []).encode(noisy_frame(), (0, 50, 50, 0))
    with pytest.raises(InvalidImageError):
        FaceEncoder(backend=lambda rgb, b: [np.zeros(7)]).encode(noisy_frame(), (0, 50, 50, 0))


def test_missing_model_library(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "face_recognition":
            raise ImportError("missing")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModelUnavailableError):
        FaceDetector(0.5).detect(noisy_frame())


# ---- recognition service ----------------------------------------------------
def build_recognition(db, settings, query_embedding, boxes=((10, 40, 40, 10),)):
    user = db.add_user_with_embeddings("S1", "Asha", None, None, [make_embedding(1)])
    detector = FaceDetector(0.5, backend=lambda rgb: list(boxes))
    encoder = FaceEncoder(backend=lambda rgb, b: [query_embedding])
    svc = RecognitionService(db, detector, encoder, FaceMatcher(settings.recognition_threshold),
                             AttendanceManager(db, settings.attendance_cooldown_seconds))
    svc.reload_embeddings()
    return user, svc


def test_known_face_identified_and_attendance_once(db, settings):
    user, svc = build_recognition(db, settings, make_embedding(1, noise=0.01))
    first = svc.process_frame(noisy_frame())[0]
    second = svc.process_frame(noisy_frame())[0]
    assert first.is_known and first.user.id == user.id and first.attendance_marked
    assert second.is_known and not second.attendance_marked
    assert len(db.query_attendance()) == 1
    assert "Asha" in first.label and "S1" in first.label


def test_unknown_face_never_marks_attendance(db, settings):
    _, svc = build_recognition(db, settings, make_embedding(99))
    result = svc.process_frame(noisy_frame())[0]
    assert not result.is_known and result.label == "Unknown" and not result.attendance_marked
    assert db.query_attendance() == []


def test_no_faces_gives_empty_result(db, settings):
    _, svc = build_recognition(db, settings, make_embedding(1), boxes=())
    assert svc.process_frame(noisy_frame()) == []


def test_deleted_biometrics_no_longer_recognised(db, settings):
    user, svc = build_recognition(db, settings, make_embedding(1))
    db.delete_biometrics(user.id)
    svc.reload_embeddings()
    assert not svc.process_frame(noisy_frame())[0].is_known


# ---- export -----------------------------------------------------------------
def test_csv_export(db, tmp_path):
    from datetime import datetime
    user = db.add_user_with_embeddings("S1", "Asha, \"A\" Rao", None, None, [make_embedding(1)])
    db.add_attendance(user.id, datetime(2025, 1, 2, 9, 30, 0), "PRESENT", 0.31234)
    out = tmp_path / "out" / "att.csv"
    assert export_attendance_csv(db.query_attendance(), out) == 1
    import pandas as pd
    df = pd.read_csv(out, encoding="utf-8-sig")
    assert list(df.columns)[:3] == ["ID", "Name", "Date"] and df.loc[0, "Name"] == 'Asha, "A" Rao'
    assert df.loc[0, "Time"] == "09:30:00" and df.loc[0, "Recognition Distance"] == 0.3123


def test_csv_export_empty_and_unwritable(tmp_path):
    assert export_attendance_csv([], tmp_path / "e.csv") == 0
    from utils.errors import AppError
    with pytest.raises(AppError):
        export_attendance_csv([], tmp_path)  # a directory is not a file


# ---- settings ---------------------------------------------------------------
def test_settings_env_override_and_validation(tmp_path):
    s = Settings.load(tmp_path / "none.json", env={"FACEAPP_RECOGNITION_THRESHOLD": "0.45", "FACEAPP_LOG_LEVEL": "debug"})
    assert s.recognition_threshold == 0.45 and s.log_level == "DEBUG"
    for bad in ({"FACEAPP_RECOGNITION_THRESHOLD": "abc"}, {"FACEAPP_FRAME_SCALE": "5"}, {"FACEAPP_LOG_LEVEL": "LOUD"}):
        with pytest.raises(ConfigError):
            Settings.load(tmp_path / "none.json", env=bad)


def test_settings_file_round_trip_and_corruption(tmp_path):
    path = tmp_path / "s.json"
    Settings(attendance_cooldown_seconds=42).save(path)
    assert Settings.load(path, env={}).attendance_cooldown_seconds == 42
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.load(path, env={})


# ---- recognition runner (background thread, fake camera) ---------------------
def test_runner_marks_attendance_then_reports_camera_loss(db, settings):
    from dataclasses import replace
    from services.recognition_service import RecognitionRunner
    _, svc = build_recognition(db, settings, make_embedding(1, noise=0.01))
    cap = FakeCapture(frames=6)
    runner = RecognitionRunner(Camera(0, lambda i: cap), svc, replace(settings, process_every_n_frames=1))
    runner.start()
    runner._thread.join(timeout=5)  # fake camera runs dry -> CameraError ends the loop
    assert not runner.is_alive
    assert runner.error and runner.status == "Camera Not Available"
    assert runner.latest_frame() is not None
    assert "attendance marked" in runner.events.get_nowait()
    assert len(db.query_attendance()) == 1  # 6 frames, one record
    runner.stop()
    assert cap.released


def test_theme_setting_validated_and_persisted(tmp_path):
    assert Settings.load(tmp_path / "none.json", env={"FACEAPP_THEME": "dark"}).theme == "dark"
    with pytest.raises(ConfigError):
        Settings.load(tmp_path / "none.json", env={"FACEAPP_THEME": "neon"})
    path = tmp_path / "s.json"
    Settings(theme="dark").save(path)
    assert Settings.load(path, env={}).theme == "dark"
