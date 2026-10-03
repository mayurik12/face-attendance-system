"""Registration window: form + live preview + sample capture."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from app_context import AppContext
from core.camera import Camera
from gui import theme
from gui.widgets import frame_to_photo
from services.registration_service import RegistrationSession, UserDetails
from utils.errors import AppError, CameraError
from utils.helpers import draw_face_box
from utils.logger import get_logger

log = get_logger(__name__)
SAMPLE_INTERVAL_S = 0.6  # pause between samples so poses/expressions vary slightly


class RegistrationWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, ctx: AppContext, on_done: Callable[[], None]) -> None:
        super().__init__(master)
        self.title("Register Person")
        self.resizable(False, False)
        theme.prepare(self)
        self._ctx, self._on_done = ctx, on_done
        self._camera: Camera | None = None
        self._session: RegistrationSession | None = None
        self._details: UserDetails | None = None
        self._box = None
        self._photo = None
        self._last_sample = 0.0
        self._after_id: str | None = None

        form = ttk.Frame(self, padding=20)
        form.grid(row=0, column=0, sticky="n")
        self._vars = {k: tk.StringVar() for k in ("id", "name", "email", "department")}
        for row, (key, label) in enumerate([("id", "ID *"), ("name", "Name *"), ("email", "Email"), ("department", "Department")]):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(form, textvariable=self._vars[key], width=28).grid(row=row, column=1, pady=4, padx=(8, 0))
        self.start_btn = ttk.Button(form, text="Start Capture", command=self._start, style="Accent.TButton")
        self.start_btn.grid(row=4, column=0, columnspan=2, pady=(12, 4), sticky="ew")
        self.progress = ttk.Progressbar(form, maximum=ctx.settings.registration_samples)
        self.progress.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        self.status = tk.StringVar(value="Fill in the form, then press Start Capture.")
        ttk.Label(form, textvariable=self.status, wraplength=260, justify="left").grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(form, text="Keep one face in view, look at the camera,\nand turn your head slightly between samples.",
                  style="Muted.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.video = ttk.Label(self, width=64, style="Video.TLabel", anchor="center", text="Camera preview")
        self.video.grid(row=0, column=1, padx=16, pady=16)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _start(self) -> None:
        try:
            self._details = self._ctx.registration.validate_details(
                self._vars["id"].get(), self._vars["name"].get(), self._vars["email"].get(), self._vars["department"].get())
            camera = Camera(self._ctx.settings.camera_index)
            camera.open()
        except AppError as exc:
            messagebox.showerror("Cannot start", str(exc), parent=self)
            return
        self._camera = camera
        self._session = self._ctx.registration.new_session()
        self.start_btn.state(["disabled"])
        self.status.set("Camera Connected. Look at the camera...")
        self._tick()

    def _tick(self) -> None:
        assert self._camera and self._session and self._details
        try:
            frame = self._camera.read()
            if time.monotonic() - self._last_sample >= SAMPLE_INTERVAL_S:
                self._last_sample = time.monotonic()
                result = self._session.process_frame(frame)
                self._box = result.box
                self.status.set(result.message)
                self.progress["value"] = len(self._session.embeddings)
            view = frame.copy()
            if self._box:
                draw_face_box(view, self._box, "Face", (60, 160, 60))
            self._photo = frame_to_photo(view, 480, 360)
            self.video.configure(image=self._photo)
            if self._session.is_complete:
                self._finish()
                return
        except CameraError as exc:
            self._abort("Camera Not Available", str(exc))
            return
        except Exception:
            log.exception("Unexpected registration error")
            self._abort("Unexpected error", "Something went wrong. Details were written to logs/app.log.")
            return
        self._after_id = self.after(30, self._tick)

    def _finish(self) -> None:
        try:
            user = self._ctx.registration.complete(self._details, self._session)
        except AppError as exc:
            self._abort("Registration failed", str(exc))
            return
        self._release()
        messagebox.showinfo("Registered", f"{user.name} ({user.employee_code}) registered successfully.", parent=self)
        self._on_done()
        self.destroy()

    def _abort(self, title: str, message: str) -> None:
        self._release()
        self.start_btn.state(["!disabled"])
        self.status.set(message)
        messagebox.showerror(title, message, parent=self)

    def _release(self) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._camera:
            self._camera.release()
            self._camera = None

    def _close(self) -> None:
        self._release()
        self.destroy()
