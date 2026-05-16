def fmt_duration(seconds: float) -> str:
    """Human-readable elapsed time: '42s', '4m 12s', or '1h 03m 17s'."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
