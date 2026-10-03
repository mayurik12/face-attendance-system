"""Application entry point and dependency wiring.

Run with:  python app.py
"""
from __future__ import annotations

import sys

from app_context import build_context
from config.settings import Settings
from utils.errors import AppError, ConfigError
from utils.logger import get_logger, setup_logging


def models_available() -> bool:
    try:
        import face_recognition  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    try:
        settings = Settings.load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(settings.log_level, settings.log_dir)
    log = get_logger("app")
    log.info("Application starting")
    try:
        context = build_context(settings)
    except AppError as exc:
        log.error("Startup failed: %s", exc)
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    from gui.dashboard import Dashboard  # imported late so tests never need Tk

    Dashboard(context, models_ok=models_available(), rebuild=build_context).mainloop()
    log.info("Application closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
