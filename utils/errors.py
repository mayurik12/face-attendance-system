"""Application-specific exceptions.

Every error that a user can reasonably trigger has its own type, so the GUI can
show a friendly message while the technical detail goes to the log file.
"""


class AppError(Exception):
    """Base class for all expected application errors."""


class ConfigError(AppError):
    """Invalid configuration value."""


class ValidationError(AppError):
    """User input failed validation."""


class DuplicateUserError(AppError):
    """A user with the same ID already exists."""


class DatabaseError(AppError):
    """Database could not be opened, queried or written."""


class CameraError(AppError):
    """Camera missing, busy, permission denied, or stopped delivering frames."""


class InvalidImageError(AppError):
    """Frame is not a usable BGR image (None, wrong shape/dtype, empty)."""


class NoFaceError(AppError):
    """No face was found in the frame."""


class MultipleFacesError(AppError):
    """More than one face was found where exactly one is required."""


class RegistrationError(AppError):
    """Registration could not be completed (e.g. too few good samples)."""


class ModelUnavailableError(AppError):
    """Face detection/embedding model or its library is not installed."""
