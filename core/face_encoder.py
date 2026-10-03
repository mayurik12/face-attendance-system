"""Face embedding extraction (feature extraction) with a pretrained model.

The dlib model is a ResNet-style CNN pretrained on ~3 million face images. It
maps an aligned face crop to a 128-number vector. The network is NOT trained
here; we only run it (inference).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from config.settings import EMBEDDING_DIM
from utils.errors import InvalidImageError, ModelUnavailableError
from utils.helpers import Box, bgr_to_rgb, ensure_valid_frame

EncoderBackend = Callable[[np.ndarray, Sequence[Box]], list[np.ndarray]]  # RGB, boxes -> embeddings


def _default_backend(num_jitters: int) -> EncoderBackend:
    try:
        import face_recognition
    except ImportError as exc:
        raise ModelUnavailableError(
            "The 'face_recognition' library (dlib) is not installed. See README -> Installation."
        ) from exc
    return lambda rgb, boxes: face_recognition.face_encodings(
        rgb, known_face_locations=list(boxes), num_jitters=num_jitters, model="small"
    )


class FaceEncoder:
    def __init__(self, backend: EncoderBackend | None = None, num_jitters: int = 1) -> None:
        self._backend = backend
        self._num_jitters = num_jitters

    def encode(self, frame: np.ndarray, box: Box) -> np.ndarray:
        """Return a float32 vector of length 128 for the face inside ``box``.

        Alignment: dlib finds 5 facial landmarks inside the box and warps the
        face to a canonical pose before the CNN runs, so no extra alignment code is needed.
        """
        ensure_valid_frame(frame)
        if self._backend is None:
            self._backend = _default_backend(self._num_jitters)
        encodings = self._backend(bgr_to_rgb(frame), [box])
        if not encodings:
            raise InvalidImageError("Could not compute a face embedding for this frame.")
        vector = np.asarray(encodings[0], dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(vector)):
            raise InvalidImageError("Face model returned an unexpected embedding.")
        return vector
