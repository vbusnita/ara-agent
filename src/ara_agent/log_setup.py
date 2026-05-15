"""Centralized logging for ara-agent.

When something goes wrong — especially a crash or hang that takes
the terminal with it — there has to be a durable trail of what was
happening just before. This module sets that up.

Log file
--------
  ~/Library/Logs/ara-agent/agent.log       (rolling: 5 MB × 5 files)
  ~/Library/Logs/ara-agent/faults.log      (faulthandler dumps SIGSEGV
                                            / SIGABRT / SIGFPE etc.)

Levels
------
  - File handler: DEBUG and above (everything goes to disk)
  - Console handler: WARNING and above by default (terminal stays
    quiet; user can opt into more with ARA_VERBOSE=1)

Hooks
-----
  - sys.excepthook        — uncaught in main thread
  - threading.excepthook  — uncaught in any background thread
  - asyncio loop handler  — installed per-loop in voice_agent
  - faulthandler          — C-level crashes (segfault, abort, etc.)

After this module's setup_logging() has run, every module just uses
`logging.getLogger(__name__)` normally. Exceptions get full tracebacks
in the file even if the process dies before they reach stdout.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path


_LOG_DIR = Path.home() / "Library" / "Logs" / "ara-agent"
_LOG_PATH = _LOG_DIR / "agent.log"
_FAULT_PATH = _LOG_DIR / "faults.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 5

_initialized = False
_fault_file = None  # kept open as long as the process lives


def log_path() -> Path:
    """Public accessor for the log file path."""
    return _LOG_PATH


def setup_logging() -> None:
    """Initialize file + console logging. Idempotent.

    Honors ARA_VERBOSE=1 env var to drop the console threshold from
    WARNING down to DEBUG (useful when you want to watch live).
    """
    global _initialized, _fault_file
    if _initialized:
        return
    _initialized = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Faulthandler must hold an open file for the lifetime of the process.
    try:
        _fault_file = open(_FAULT_PATH, "a", buffering=1)
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        pass  # if we can't open it we still want regular logging

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_PATH,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=("%(asctime)s.%(msecs)03d [%(levelname)-5s] "
             "%(name)s:%(funcName)s:%(lineno)d  %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    console_level = (
        logging.DEBUG if os.environ.get("ARA_VERBOSE") == "1"
        else logging.WARNING
    )
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(levelname)s %(name)s: %(message)s",
    ))
    root.addHandler(console_handler)

    _install_exception_hooks()

    logging.getLogger("ara_agent").info(
        "logging initialized → %s (faults: %s)", _LOG_PATH, _FAULT_PATH,
    )


def install_asyncio_hook(loop) -> None:
    """Route asyncio's default exception handler through our logger.
    Call once per loop, immediately after creating it."""
    log = logging.getLogger("ara_agent.asyncio")

    def handler(_loop, context):
        msg = context.get("message", "unknown asyncio error")
        exc = context.get("exception")
        if exc is not None:
            log.error(
                "asyncio: %s",
                msg,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            # Build a useful summary from the context dict (task, future, etc.)
            extras = {k: v for k, v in context.items() if k != "message"}
            log.error("asyncio: %s — %s", msg, extras)

    loop.set_exception_handler(handler)


def _install_exception_hooks() -> None:
    log = logging.getLogger("ara_agent.crash")

    def sys_hook(exc_type, exc_value, exc_tb):
        # Let ⌃C still produce the usual KeyboardInterrupt traceback.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical(
            "Uncaught exception in main thread:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = sys_hook

    def thread_hook(args):
        thread_name = args.thread.name if args.thread else "<unknown>"
        log.critical(
            "Uncaught exception in thread %r:\n%s",
            thread_name,
            "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback,
            )),
        )

    threading.excepthook = thread_hook
