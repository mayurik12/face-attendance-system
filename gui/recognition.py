"""Live recognition panel embedded in the dashboard."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from app_context import AppContext
from core.camera import Camera
from gui import theme
from gui.widgets import frame_to_photo
from services.recognition_service import RecognitionRunner
from utils.errors import AppError
from utils.logger import get_logger

log = get_logger(__name__)
IDLE_TEXT = "Camera is off\n\nPress  ▶ Start Recognition  to begin"


class RecognitionPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, get_context: Callable[[], AppContext], set_status: Callable[[str], None]) -> None:
        super().__init__(master)
        self._get_context, self._set_status = get_context, set_status
        self._runner: RecognitionRunner | None = None
        self._photo = None
        self._after_id: str | None = None
        video_card = ttk.Frame(self, style="Card.TFrame", padding=10)
        video_card.pack(fill="both", expand=True)
        self.video = ttk.Label(video_card, style="Video.TLabel", anchor="center", justify="center", text=IDLE_TEXT)
        self.video.pack(fill="both", expand=True)
        events_card = ttk.Frame(self, style="Card.TFrame", padding=(16, 12))
        events_card.pack(fill="x", pady=(14, 0))
        ttk.Label(events_card, text="Recent attendance", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.events = tk.Listbox(events_card, height=4)
        self.events.pack(fill="x")
        theme.apply_theme(self.winfo_toplevel(), theme.current_mode())

    @property
    def running(self) -> bool:
        return self._runner is not None

    def start(self) -> None:
        if self._runner:
            return
        ctx = self._get_context()
        try:
            ctx.recognition.reload_embeddings()
            if ctx.matcher.sample_count == 0:
                messagebox.showinfo("No registered faces", "Register at least one person first. All faces will show as Unknown until then.")
            runner = RecognitionRunner(Camera(ctx.settings.camera_index), ctx.recognition, ctx.settings)
            runner.start()
        except AppError as exc:
            self._set_status("Camera Not Available" if "amera" in str(exc) else "Error")
            messagebox.showerror("Cannot start recognition", str(exc))
            return
        self._runner = runner
        self._set_status("Camera Connected")
        self._poll()

    def stop(self) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._runner:
            self._runner.stop()
            self._runner = None
        self.video.configure(image="", text=IDLE_TEXT)
        self._photo = None
        self._set_status("Recognition stopped")

    def _poll(self) -> None:
        runner = self._runner
        if runner is None:
            return
        frame = runner.latest_frame()
        if frame is not None:
            self._photo = frame_to_photo(frame, max(self.video.winfo_width(), 320), max(self.video.winfo_height(), 240))
            self.video.configure(image=self._photo, text="")
        while not runner.events.empty():
            self.events.insert(0, runner.events.get_nowait())
        if runner.error:
            message = runner.error
            self.stop()
            self._set_status("Camera Not Available")
            messagebox.showerror("Recognition stopped", message)
            return
        self._set_status(runner.status)
        self._after_id = self.after(33, self._poll)
