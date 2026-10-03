"""Plain data containers returned by the database layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: int
    employee_code: str
    name: str
    email: str | None
    department: str | None
    created_at: str
