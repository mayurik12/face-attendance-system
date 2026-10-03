"""Settings dialog. Values are validated before they are saved."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from config.settings import Settings
from gui import theme
from utils.errors import ConfigError

FIELDS = [
    ("recognition_threshold", "Recognition threshold (lower = stricter)"),
    ("registration_samples", "Registration samples"),
    ("camera_index", "Camera index"),
    ("frame_scale", "Frame scale (0.1 - 1.0)"),
    ("process_every_n_frames", "Process every Nth frame"),
    ("attendance_cooldown_seconds", "Attendance cooldown (seconds)"),
]


class SettingsDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, settings: Settings, on_save: Callable[[Settings], None]) -> None:
        super().__init__(master)
        self.title("Settings")
        self.resizable(False, False)
        theme.prepare(self)
        self._settings, self._on_save = settings, on_save
        body = ttk.Frame(self, padding=22)
        body.pack()
        self._vars = {}
        for row, (key, label) in enumerate(FIELDS):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=4)
            self._vars[key] = tk.StringVar(value=str(getattr(settings, key)))
            ttk.Entry(body, textvariable=self._vars[key], width=10).grid(row=row, column=1, padx=(10, 0))
        ttk.Button(body, text="Save", command=self._save, style="Accent.TButton").grid(row=len(FIELDS), column=0, columnspan=2, pady=(16, 0), sticky="ew")

    def _save(self) -> None:
        try:
            base = self._settings.to_json_dict()
            base.update({k: v.get() for k, v in self._vars.items()})
            new = Settings.from_mapping(base)
            new.save()
        except (ConfigError, OSError) as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self)
            return
        self._on_save(new)
        self.destroy()
