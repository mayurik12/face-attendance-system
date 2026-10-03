"""Webcam wrapper with an explicit lifecycle and friendly errors."""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from utils.errors import CameraError
from utils.logger import get_logger

log = get_logger(__name__)


def _default_factory(index: int) -> Any:
    import cv2

    # DirectShow opens much faster than the default backend on Windows.
    return cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)


class Camera:
    def __init__(self, index: int = 0, capture_factory: Callable[[int], Any] | None = None) -> None:
        self.index = index
        self._factory = capture_factory or _default_factory
        self._capture: Any = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    def open(self) -> None:
        if self._capture is not None:
            return
        try:
            capture = self._factory(self.index)
        except Exception as exc:  # backend-specific failures
            log.error("Camera %s could not be created: %s", self.index, exc)
            raise CameraError("Camera not available. Check that it is connected and allowed in system privacy settings.") from exc
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            log.error("Camera %s failed to open", self.index)
            raise CameraError("Camera not available. Close other apps using it, check the camera index and Windows camera permissions.")
        self._capture = capture
        log.info("Camera %s opened", self.index)

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is not open.")
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            log.error("Camera read failed: %s", exc)
            raise CameraError("Lost connection to the camera.") from exc
        if not ok or frame is None:
            raise CameraError("Camera did not return a frame (disconnected or in use).")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None
                log.info("Camera %s released", self.index)

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
