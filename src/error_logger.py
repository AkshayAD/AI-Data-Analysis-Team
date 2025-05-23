import os
import traceback
from datetime import datetime

LOG_FILE = "crash_log.txt"


def log_exception(exc: Exception) -> None:
    """Append exception info and traceback to the crash log."""
    timestamp = datetime.utcnow().isoformat()
    trace = traceback.format_exc()
    entry = f"{timestamp} - {exc}\n{trace}\n\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def read_log() -> str:
    """Return contents of the crash log if it exists."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def clear_log() -> None:
    """Delete the crash log file."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
