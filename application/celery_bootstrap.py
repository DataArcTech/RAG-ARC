import threading

from dotenv import load_dotenv

import app_registration

_lock = threading.Lock()
_initialized = False


def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        load_dotenv()
        app_registration.initialize()
        _initialized = True

