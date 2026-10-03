"""CSV export of attendance records."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.errors import AppError
from utils.logger import get_logger

log = get_logger(__name__)
COLUMNS = {
    "employee_code": "ID", "name": "Name", "date": "Date", "time": "Time",
    "status": "Status", "recognition_distance": "Recognition Distance",
}


def export_attendance_csv(rows: list[dict], path: Path | str) -> int:
    """Write rows to CSV (UTF-8 with BOM so Excel opens it correctly). Returns row count."""
    frame = pd.DataFrame(rows, columns=list(COLUMNS)).rename(columns=COLUMNS)
    if "Recognition Distance" in frame:
        frame["Recognition Distance"] = frame["Recognition Distance"].round(4)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    except OSError as exc:
        log.error("CSV export failed: %s", exc)
        raise AppError(f"Could not write the file: {exc.strerror or exc}") from exc
    log.info("Exported %d attendance rows", len(frame))
    return len(frame)
