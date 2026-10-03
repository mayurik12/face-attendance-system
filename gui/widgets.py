"""Shared Tk helpers."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageTk


def frame_to_photo(frame_bgr: np.ndarray, max_width: int, max_height: int) -> ImageTk.PhotoImage:
    """Convert an OpenCV frame to a Tk image that fits inside the given box."""
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image.thumbnail((max_width, max_height))
    return ImageTk.PhotoImage(image)
