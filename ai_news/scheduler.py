"""Background refresh loop. Runs while the Space is awake (free Spaces sleep after inactivity;
the next visitor wakes it and the corpus is pulled from the dataset repo again)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


def start(refresh_fn: Callable[[], str], every_hours: float, run_first_after: float | None = 30.0) -> threading.Thread:
    def loop() -> None:
        delay = run_first_after if run_first_after is not None else every_hours * 3600
        while True:
            time.sleep(delay)
            try:
                refresh_fn()
            except Exception:
                log.exception("scheduled refresh failed")
            delay = every_hours * 3600

    t = threading.Thread(target=loop, name="corpus-refresh", daemon=True)
    t.start()
    return t
