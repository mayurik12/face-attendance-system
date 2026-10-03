"""Attendance history with filters and CSV export."""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from app_context import AppContext
from gui import theme
from services.export_service import export_attendance_csv
from utils.errors import AppError
from utils.helpers import parse_date

COLUMNS = [("employee_code", "ID", 90), ("name", "Name", 180), ("date", "Date", 100),
           ("time", "Time", 80), ("status", "Status", 80), ("recognition_distance", "Distance", 80)]


class AttendanceWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, ctx: AppContext) -> None:
        super().__init__(master)
        self.title("Attendance Records")
        self.geometry("1120x500")
        self.minsize(1100, 320)
        theme.prepare(self)
        self._ctx = ctx
        self._rows: list[dict] = []
        bar = ttk.Frame(self, padding=(16, 14, 16, 10))
        bar.pack(fill="x")
        self._vars = {k: tk.StringVar() for k in ("name", "id", "date", "from", "to")}
        for col, (key, label, width) in enumerate([("name", "Name", 16), ("id", "ID", 10), ("date", "Date (YYYY-MM-DD)", 12),
                                                   ("from", "From", 12), ("to", "To", 12)]):
            ttk.Label(bar, text=label).grid(row=0, column=col, sticky="w", padx=3)
            entry = ttk.Entry(bar, textvariable=self._vars[key], width=width)
            entry.grid(row=1, column=col, padx=3)
            entry.bind("<Return>", lambda _e: self.search())
            if key == "name":
                entry.focus_set()
        ttk.Button(bar, text="Search", command=self.search, style="Accent.TButton").grid(row=1, column=5, padx=(10, 4))
        ttk.Button(bar, text="Clear", command=self._clear).grid(row=1, column=6)
        ttk.Button(bar, text="Export CSV", command=self.export).grid(row=1, column=7, padx=4)
        self.tree = ttk.Treeview(self, columns=[c[0] for c in COLUMNS], show="headings")
        for key, label, width in COLUMNS:
            self.tree.heading(key, text=label, anchor="w")
            self.tree.column(key, width=width, anchor="w")
        theme.stripe(self.tree)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.count = tk.StringVar()
        ttk.Label(self, textvariable=self.count, style="Muted.TLabel").pack(anchor="w", padx=16, pady=8)
        self.search()

    def _clear(self) -> None:
        for var in self._vars.values():
            var.set("")
        self.search()

    def search(self) -> None:
        try:
            self._rows = self._ctx.db.query_attendance(
                name=self._vars["name"].get(), employee_code=self._vars["id"].get(),
                on_date=parse_date(self._vars["date"].get(), "Date"),
                date_from=parse_date(self._vars["from"].get(), "From"),
                date_to=parse_date(self._vars["to"].get(), "To"))
        except AppError as exc:
            messagebox.showerror("Search", str(exc), parent=self)
            return
        self.tree.delete(*self.tree.get_children())
        for i, row in enumerate(self._rows):
            dist = row["recognition_distance"]
            self.tree.insert("", "end", values=[row["employee_code"], row["name"], row["date"], row["time"],
                                                row["status"], "" if dist is None else f"{dist:.3f}"],
                             tags=("odd",) if i % 2 else ())
        self.count.set(f"{len(self._rows)} record(s)")

    def export(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{datetime.now():%Y%m%d_%H%M}.csv")
        if not path:
            return
        try:
            count = export_attendance_csv(self._rows, path)
        except AppError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
            return
        messagebox.showinfo("Export complete", f"{count} record(s) exported.", parent=self)
