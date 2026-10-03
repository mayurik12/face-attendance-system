"""Modern light/dark theme built on ttk's 'clam' engine (no extra dependencies)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

FONT = "Segoe UI"  # Windows UI font; Tk falls back to the system sans-serif elsewhere

PALETTES: dict[str, dict[str, str]] = {
    "light": dict(bg="#F3F5F9", surface="#FFFFFF", sidebar="#0F172A", sidebar_hover="#1E293B", sidebar_text="#E2E8F0",
                  text="#0F172A", muted="#64748B", border="#E2E8F0", accent="#4F46E5", accent_hover="#4338CA",
                  danger="#E11D48", danger_hover="#BE123C", success="#16A34A", warning="#D97706",
                  field="#FFFFFF", stripe="#F8FAFC", header="#EEF2F7", video="#0B1020"),
    "dark": dict(bg="#0B1120", surface="#111827", sidebar="#070B16", sidebar_hover="#172033", sidebar_text="#CBD5E1",
                 text="#E5E7EB", muted="#94A3B8", border="#243044", accent="#6366F1", accent_hover="#818CF8",
                 danger="#F43F5E", danger_hover="#FB7185", success="#22C55E", warning="#F59E0B",
                 field="#0F172A", stripe="#0F1729", header="#172033", video="#05080F"),
}
_current = "light"


def current_mode() -> str:
    return _current


def palette() -> dict[str, str]:
    return PALETTES[_current]


def apply_theme(root: tk.Misc, mode: str = "light") -> dict[str, str]:
    """(Re)style every ttk widget and recolour existing classic-Tk widgets."""
    global _current
    _current = mode if mode in PALETTES else "light"
    p = PALETTES[_current]
    s = ttk.Style(root)
    s.theme_use("clam")
    base = (FONT, 10)

    s.configure(".", background=p["bg"], foreground=p["text"], font=base, bordercolor=p["border"],
                lightcolor=p["border"], darkcolor=p["border"], focuscolor=p["bg"], troughcolor=p["header"])
    for name, bg in (("TFrame", p["bg"]), ("Card.TFrame", p["surface"]), ("Sidebar.TFrame", p["sidebar"])):
        s.configure(name, background=bg)
    s.configure("Card.TFrame", relief="solid", borderwidth=1, bordercolor=p["border"],
                lightcolor=p["border"], darkcolor=p["border"])

    s.configure("TLabel", background=p["bg"], foreground=p["text"])
    s.configure("Card.TLabel", background=p["surface"])
    s.configure("Title.TLabel", font=(FONT, 20, "bold"))
    s.configure("Section.TLabel", font=(FONT, 11, "bold"), background=p["surface"])
    s.configure("Muted.TLabel", foreground=p["muted"])
    s.configure("CardMuted.TLabel", foreground=p["muted"], background=p["surface"])
    s.configure("Brand.TLabel", background=p["sidebar"], foreground="#FFFFFF", font=(FONT, 15, "bold"))
    s.configure("SideMuted.TLabel", background=p["sidebar"], foreground="#64748B", font=(FONT, 8, "bold"))
    s.configure("Video.TLabel", background=p["video"], foreground="#94A3B8", font=(FONT, 11), anchor="center")
    for name, color in (("ChipOk", p["success"]), ("ChipBad", p["danger"]), ("ChipWarn", p["warning"]), ("ChipNeutral", p["muted"])):
        s.configure(f"{name}.TLabel", background=p["surface"], foreground=color, font=(FONT, 10, "bold"),
                    padding=(14, 7), relief="solid", borderwidth=1, bordercolor=p["border"])

    s.configure("TButton", background=p["surface"], foreground=p["text"], bordercolor=p["border"], borderwidth=1,
                relief="flat", padding=(14, 8), focusthickness=0)
    s.map("TButton", background=[("active", p["header"]), ("disabled", p["bg"])], foreground=[("disabled", p["muted"])])
    for name, bg, hover in (("Accent", p["accent"], p["accent_hover"]), ("Danger", p["danger"], p["danger_hover"])):
        s.configure(f"{name}.TButton", background=bg, foreground="#FFFFFF", bordercolor=bg, padding=(16, 10),
                    font=(FONT, 10, "bold"))
        s.map(f"{name}.TButton", background=[("active", hover), ("disabled", p["border"])], bordercolor=[("active", hover)])
    s.configure("Sidebar.TButton", background=p["sidebar"], foreground=p["sidebar_text"], borderwidth=0,
                bordercolor=p["sidebar"], anchor="w", padding=(18, 10))
    s.map("Sidebar.TButton", background=[("active", p["sidebar_hover"])], bordercolor=[("active", p["sidebar_hover"])],
          foreground=[("active", "#FFFFFF")])

    s.configure("TEntry", fieldbackground=p["field"], foreground=p["text"], insertcolor=p["text"], padding=7,
                bordercolor=p["border"], lightcolor=p["border"], darkcolor=p["border"])
    s.map("TEntry", bordercolor=[("focus", p["accent"])], lightcolor=[("focus", p["accent"])], darkcolor=[("focus", p["accent"])])

    s.configure("Treeview", background=p["surface"], fieldbackground=p["surface"], foreground=p["text"], rowheight=32,
                borderwidth=0, font=base)
    s.configure("Treeview.Heading", background=p["header"], foreground=p["muted"], relief="flat", padding=(8, 8),
                font=(FONT, 9, "bold"))
    s.map("Treeview.Heading", background=[("active", p["border"])])
    s.map("Treeview", background=[("selected", p["accent"])], foreground=[("selected", "#FFFFFF")])

    s.configure("Horizontal.TProgressbar", troughcolor=p["header"], background=p["accent"], bordercolor=p["header"],
                lightcolor=p["accent"], darkcolor=p["accent"], thickness=10)
    s.configure("Vertical.TScrollbar", background=p["border"], troughcolor=p["bg"], bordercolor=p["bg"],
                arrowcolor=p["muted"], relief="flat")

    _recolor(root, p)
    return p


def prepare(window: tk.Misc) -> None:
    """Give a new Toplevel the current background."""
    window.configure(background=palette()["bg"])


def stripe(tree: ttk.Treeview) -> None:
    tree.tag_configure("odd", background=palette()["stripe"])


def chip_style(text: str) -> str:
    t = text.lower()
    if "stopped" in t or t in {"ready", ""}:
        return "ChipNeutral.TLabel"
    if any(k in t for k in ("not available", "error", "missing", "fail", "not installed")):
        return "ChipBad.TLabel"
    if "unknown" in t:
        return "ChipWarn.TLabel"
    if any(k in t for k in ("marked", "connected", "registered", "saved", "recogni", "detected")):
        return "ChipOk.TLabel"
    return "ChipNeutral.TLabel"


def _recolor(widget: tk.Misc, p: dict[str, str]) -> None:
    if isinstance(widget, (tk.Tk, tk.Toplevel)):
        widget.configure(background=p["bg"])
    elif isinstance(widget, tk.Listbox):
        widget.configure(background=p["surface"], foreground=p["text"], selectbackground=p["accent"],
                         selectforeground="#FFFFFF", highlightthickness=0, borderwidth=0, relief="flat",
                         font=(FONT, 10), activestyle="none")
    elif isinstance(widget, ttk.Treeview):
        widget.tag_configure("odd", background=p["stripe"])
    for child in widget.winfo_children():
        _recolor(child, p)
