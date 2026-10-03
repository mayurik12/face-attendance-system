"""Administrator view: list users, delete user, or delete only biometric data."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from app_context import AppContext
from gui import theme
from utils.errors import AppError


class UserManagementWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, ctx: AppContext, on_change: Callable[[], None]) -> None:
        super().__init__(master)
        self.title("Manage Users")
        self.geometry("760x400")
        self.minsize(760, 240)
        theme.prepare(self)
        self._ctx, self._on_change = ctx, on_change
        self._users: dict[str, object] = {}
        cols = ("code", "name", "email", "department", "samples")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        for key, label, width in [("code", "ID", 90), ("name", "Name", 170), ("email", "Email", 170),
                                  ("department", "Department", 150), ("samples", "Face samples", 110)]:
            self.tree.heading(key, text=label, anchor="w")
            self.tree.column(key, width=width)
        bar = ttk.Frame(self, padding=(16, 0, 16, 16))
        bar.pack(fill="x", side="bottom")  # packed first so it is never pushed out of view
        theme.stripe(self.tree)
        self.tree.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Button(bar, text="Delete User", command=self._delete_user, style="Danger.TButton").pack(side="left")
        ttk.Button(bar, text="Delete Face Data Only", command=self._delete_biometrics).pack(side="left", padx=8)
        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._users.clear()
        try:
            for i, user in enumerate(self._ctx.db.list_users()):
                item = self.tree.insert("", "end", tags=("odd",) if i % 2 else (), values=(user.employee_code, user.name, user.email or "",
                                                           user.department or "", self._ctx.db.count_embeddings(user.id)))
                self._users[item] = user
        except AppError as exc:
            messagebox.showerror("Manage Users", str(exc), parent=self)

    def _selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Manage Users", "Select a user first.", parent=self)
            return None
        return self._users[selection[0]]

    def _delete_user(self) -> None:
        user = self._selected()
        if user and messagebox.askyesno(
                "Delete user", f"Permanently delete {user.name} ({user.employee_code}), including face data AND attendance history?",
                parent=self):
            self._run(lambda: self._ctx.db.delete_user(user.id))

    def _delete_biometrics(self) -> None:
        user = self._selected()
        if user and messagebox.askyesno(
                "Delete face data", f"Delete all stored face data for {user.name}? They will no longer be recognised, "
                "but the user record and attendance history are kept.", parent=self):
            self._run(lambda: self._ctx.db.delete_biometrics(user.id))

    def _run(self, action: Callable[[], object]) -> None:
        try:
            action()
        except AppError as exc:
            messagebox.showerror("Manage Users", str(exc), parent=self)
            return
        self.refresh()
        self._on_change()
