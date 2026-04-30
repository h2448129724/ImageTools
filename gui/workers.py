"""Background worker threads for long-running operations."""
import inspect
import threading
import traceback as _traceback
from PySide6.QtCore import QThread, Signal

from utils.exceptions import ImageToolsError


class WorkerThread(QThread):
    """Base worker that runs a function in a background thread.

    Signals:
        finished(result): emitted when the function completes successfully.
        progress(current, total): emitted for progress updates.
        error(user_msg, full_traceback): emitted when the function raises.
        log(msg): emitted for log messages from the worker.
    """
    finished = Signal(object)
    progress = Signal(int, int)
    error = Signal(str, str)
    log = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = threading.Event()

    def run(self):
        try:
            sig = inspect.signature(self.func)
            if "progress_callback" in sig.parameters:
                self.kwargs["progress_callback"] = self._on_progress
            if "cancel_check" in sig.parameters:
                self.kwargs["cancel_check"] = self.is_cancelled
            result = self.func(*self.args, **self.kwargs)
            if not self._cancelled.is_set():
                self.finished.emit(result)
        except ImageToolsError as e:
            # Business-level error: show a concise message, no need for a full traceback
            self.error.emit(str(e), "")
        except Exception as e:
            tb = _traceback.format_exc()
            self.error.emit(str(e), tb)

    def cancel(self):
        self._cancelled.set()

    def is_cancelled(self):
        return self._cancelled.is_set()

    def _on_progress(self, current, total):
        self.progress.emit(current, total)
