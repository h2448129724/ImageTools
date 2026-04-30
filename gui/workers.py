"""Background worker threads for long-running operations."""
import inspect
from PySide6.QtCore import QThread, Signal


class WorkerThread(QThread):
    """Base worker that runs a function in a background thread."""
    finished = Signal(object)
    progress = Signal(int, int)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False

    def run(self):
        try:
            sig = inspect.signature(self.func)
            if "progress_callback" in sig.parameters:
                self.kwargs["progress_callback"] = self._on_progress
            result = self.func(*self.args, **self.kwargs)
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def cancel(self):
        self._cancelled = True

    def _on_progress(self, current, total):
        self.progress.emit(current, total)
