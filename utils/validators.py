"""Input validation for registration and filters. All functions return cleaned values."""
from __future__ import annotations

import re

from utils.errors import ValidationError

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,98}[A-Za-z.]$|^[A-Za-z]$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{1,31}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_name(name: str | None) -> str:
    cleaned = " ".join((name or "").split())
    if not cleaned:
        raise ValidationError("Name must not be empty.")
    if not _NAME_RE.match(cleaned):
        raise ValidationError("Name may contain only letters, spaces, dots, apostrophes and hyphens (max 100 chars).")
    return cleaned


def validate_user_code(code: str | None) -> str:
    cleaned = (code or "").strip()
    if not cleaned:
        raise ValidationError("Employee/Student ID must not be empty.")
    if not _ID_RE.match(cleaned):
        raise ValidationError("ID must be 2-32 characters: letters, digits, '_' or '-'.")
    return cleaned


def validate_email(email: str | None) -> str | None:
    cleaned = (email or "").strip()
    if not cleaned:
        return None
    if len(cleaned) > 254 or not _EMAIL_RE.match(cleaned):
        raise ValidationError("Email address is not valid.")
    return cleaned


def validate_department(department: str | None) -> str | None:
    cleaned = " ".join((department or "").split())
    if not cleaned:
        return None
    if len(cleaned) > 100:
        raise ValidationError("Department must be at most 100 characters.")
    return cleaned
