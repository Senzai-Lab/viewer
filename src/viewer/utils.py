from __future__ import annotations


def format_bytes(nbytes: int | None) -> str:
    if nbytes is None:
        return "unbounded"

    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"