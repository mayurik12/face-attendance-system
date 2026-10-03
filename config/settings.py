"""Centralised, validated configuration.

Precedence (lowest -> highest): defaults < data/user_settings.json < environment
variables (prefix ``FACEAPP_``, e.g. ``FACEAPP_RECOGNITION_THRESHOLD=0.45``).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from utils.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIM = 128  # dlib ResNet face model output size
ENV_PREFIX = "FACEAPP_"
USER_SETTINGS_FILE = BASE_DIR / "data" / "user_settings.json"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class Settings:
    recognition_threshold: float = 0.5      # max Euclidean distance for a match
    registration_samples: int = 10          # good samples needed to register
    camera_index: int = 0
    frame_scale: float = 0.25               # detection runs on a scaled-down frame
    process_every_n_frames: int = 3         # recognition runs on every Nth frame
    attendance_cooldown_seconds: int = 300  # no repeat attendance within this
    min_face_size: int = 80                 # px, in the full-resolution frame
    min_sharpness: float = 40.0             # Laplacian variance; lower = blurry
    db_path: Path = BASE_DIR / "data" / "attendance.db"
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"
    theme: str = "light"                    # "light" or "dark"

    def validate(self) -> "Settings":
        """Return self if valid, otherwise raise ConfigError."""
        checks = [
            (0.1 <= self.recognition_threshold <= 1.2, "recognition_threshold must be between 0.1 and 1.2"),
            (3 <= self.registration_samples <= 50, "registration_samples must be between 3 and 50"),
            (self.camera_index >= 0, "camera_index must be >= 0"),
            (0.1 <= self.frame_scale <= 1.0, "frame_scale must be between 0.1 and 1.0"),
            (1 <= self.process_every_n_frames <= 30, "process_every_n_frames must be between 1 and 30"),
            (self.attendance_cooldown_seconds >= 0, "attendance_cooldown_seconds must be >= 0"),
            (self.min_face_size >= 20, "min_face_size must be >= 20"),
            (self.min_sharpness >= 0, "min_sharpness must be >= 0"),
            (self.theme in ("light", "dark"), "theme must be 'light' or 'dark'"),
            (self.log_level.upper() in VALID_LOG_LEVELS, f"log_level must be one of {sorted(VALID_LOG_LEVELS)}"),
        ]
        for ok, message in checks:
            if not ok:
                raise ConfigError(message)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["db_path"] = str(self.db_path)
        data["log_dir"] = str(self.log_dir)
        return data

    def save(self, path: Path = USER_SETTINGS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = USER_SETTINGS_FILE, env: dict[str, str] | None = None) -> "Settings":
        """Load settings from file + environment and validate them."""
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        if path.exists():
            try:
                values.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"Cannot read settings file {path}: {exc}") from exc
        for key, value in env.items():
            if key.startswith(ENV_PREFIX):
                values[key[len(ENV_PREFIX):].lower()] = value
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Settings":
        defaults = cls()
        known = {f.name: f for f in fields(cls)}
        converted: dict[str, Any] = {}
        for key, raw in values.items():
            if key not in known:
                continue  # ignore unknown keys instead of crashing
            kind = type(getattr(defaults, key))
            try:
                if kind is Path:
                    converted[key] = Path(str(raw))
                elif kind is str:
                    converted[key] = str(raw).upper() if key == "log_level" else str(raw)
                else:
                    converted[key] = kind(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"Invalid value for {key}: {raw!r}") from exc
        return replace(defaults, **converted).validate()
