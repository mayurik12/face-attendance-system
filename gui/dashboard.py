"""Main window: dark sidebar + content area with live recognition."""
from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from app_context import AppContext
from config.settings import Settings
from gui import theme
from gui.attendance_view import AttendanceWindow
from gui.recognition import RecognitionPanel
from gui.registration import RegistrationWindow
from gui.settings_dialog import SettingsDialog
from gui.user_management import UserManagementWindow
from services.export_service import export_attendance_csv
from utils.errors import AppError
from utils.logger import get_logger

log = get_logger(__name__)
NOTICE = "For authorised users in controlled environments only. Face recognition can make false matches and misses."


class Dashboard(tk.Tk):
    def __init__(self, ctx: AppContext, models_ok: bool, rebuild: Callable[[Settings], AppContext]) -> None:
        super().__init__()
        self.title("FaceAttend - Face Recognition Attendance System")
        self.geometry("1120x700")
        self.minsize(960, 600)
        self._ctx, self._rebuild = ctx, rebuild
        theme.apply_theme(self, ctx.settings.theme)
        self.status = tk.StringVar(value="Ready")

        # ---- sidebar -------------------------------------------------------------------
        side = ttk.Frame(self, style="Sidebar.TFrame", width=240)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="FaceAttend", style="Brand.TLabel", padding=(20, 22, 20, 0)).pack(anchor="w")
        ttk.Label(side, text="ATTENDANCE SYSTEM", style="SideMuted.TLabel", padding=(20, 2, 20, 14)).pack(anchor="w")

        def section(title: str) -> None:
            ttk.Label(side, text=title, style="SideMuted.TLabel", padding=(20, 14, 0, 4)).pack(anchor="w")

        def button(text: str, command: Callable[[], None], style: str = "Sidebar.TButton") -> ttk.Button:
            btn = ttk.Button(side, text=text, command=command, style=style)
            btn.pack(fill="x", padx=12 if style != "Sidebar.TButton" else 8, pady=2)
            return btn

        section("RECOGNITION")
        button("▶  Start Recognition", lambda: self.panel.start(), "Accent.TButton")
        button("■  Stop Recognition", lambda: self.panel.stop(), "Danger.TButton")
        section("PEOPLE")
        button("Register Person", self.open_registration)
        button("Manage Users", self.open_users)
        section("RECORDS")
        button("Attendance Records", self.open_attendance)
        button("Search Attendance", self.open_attendance)
        button("Export CSV", self.export_all)
        section("SYSTEM")
        button("Settings", self.open_settings)
        self.theme_btn = button("", self.toggle_theme)
        button("Exit", self.exit_app)
        self._refresh_theme_button()

        # ---- content --------------------------------------------------------------------
        content = ttk.Frame(self, padding=(26, 22, 26, 14))
        content.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(content)
        header.pack(fill="x")
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Live Recognition", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Known faces mark attendance automatically.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        self.chip = ttk.Label(header, textvariable=self.status, style="ChipNeutral.TLabel")
        self.chip.pack(side="right")
        self.status.trace_add("write", lambda *_: self.chip.configure(style=theme.chip_style(self.status.get())))

        ttk.Label(content, text=NOTICE, style="Muted.TLabel", padding=(0, 8, 0, 0)).pack(side="bottom", anchor="w")
        self.panel = RecognitionPanel(content, lambda: self._ctx, self.status.set)
        self.panel.pack(fill="both", expand=True, pady=(16, 0))

        self.protocol("WM_DELETE_WINDOW", self.exit_app)
        if not models_ok:
            self.status.set("Face library not installed")
            messagebox.showwarning("Face library missing",
                                   "The 'face_recognition' (dlib) library is not installed, so registration and recognition are disabled.\n"
                                   "Attendance records can still be viewed. See README -> Installation.")

    # ---- theme ------------------------------------------------------------------------
    def _refresh_theme_button(self) -> None:
        self.theme_btn.configure(text="Dark mode" if theme.current_mode() == "light" else "Light mode")

    def toggle_theme(self) -> None:
        mode = "dark" if self._ctx.settings.theme == "light" else "light"
        self._ctx.settings = replace(self._ctx.settings, theme=mode)
        try:
            self._ctx.settings.save()
        except OSError as exc:
            log.warning("Could not save theme preference: %s", exc)
        theme.apply_theme(self, mode)
        self.chip.configure(style=theme.chip_style(self.status.get()))
        self._refresh_theme_button()

    # ---- misc -------------------------------------------------------------------------
    def report_callback_exception(self, exc, val, tb) -> None:  # Tk callback errors must not kill the app
        log.error("Unhandled GUI error", exc_info=(exc, val, tb))
        messagebox.showerror("Unexpected error", "Something went wrong. Details were written to logs/app.log.")

    def _reload_cache(self) -> None:
        if self.panel.running:
            try:
                self._ctx.recognition.reload_embeddings()
            except AppError as exc:
                messagebox.showerror("Error", str(exc))

    def open_registration(self) -> None:
        self.panel.stop()  # the webcam can only be used by one component at a time
        RegistrationWindow(self, self._ctx, on_done=lambda: self.status.set("Person registered"))

    def open_attendance(self) -> None:
        AttendanceWindow(self, self._ctx)

    def open_users(self) -> None:
        UserManagementWindow(self, self._ctx, on_change=self._reload_cache)

    def open_settings(self) -> None:
        SettingsDialog(self, self._ctx.settings, self._apply_settings)

    def _apply_settings(self, settings: Settings) -> None:
        self.panel.stop()
        try:
            self._ctx = self._rebuild(settings)
        except AppError as exc:
            messagebox.showerror("Settings", str(exc))
            return
        self.status.set("Settings saved and applied")

    def export_all(self) -> None:
        try:
            rows = self._ctx.db.query_attendance()
            if not rows:
                messagebox.showinfo("Export", "There are no attendance records to export.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if path:
                count = export_attendance_csv(rows, path)
                messagebox.showinfo("Export complete", f"{count} record(s) exported.")
        except AppError as exc:
            messagebox.showerror("Export failed", str(exc))

    def exit_app(self) -> None:
        self.panel.stop()
        self.destroy()
