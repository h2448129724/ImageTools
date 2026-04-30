"""Stepper navigation bar for guiding users through the workflow."""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics


class StepperBar(QWidget):
    """Horizontal stepper with 5 steps: input → function → params → preview → run."""

    STEP_TITLES = ["选择输入", "选择功能", "配置参数", "预览确认", "执行处理"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self._completed = set()
        self._errors = set()
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_step(self, index: int):
        """Set the currently active step (0-based)."""
        self._current = max(0, min(index, len(self.STEP_TITLES) - 1))
        self.update()

    def mark_complete(self, index: int):
        """Mark a step as completed."""
        self._completed.add(index)
        self._errors.discard(index)
        self.update()

    def mark_error(self, index: int):
        """Mark a step as having an error."""
        self._errors.add(index)
        self.update()

    def clear_error(self, index: int):
        """Clear error state from a step."""
        self._errors.discard(index)
        self.update()

    def reset(self):
        """Reset all progress."""
        self._current = 0
        self._completed.clear()
        self._errors.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        n = len(self.STEP_TITLES)
        step_w = width / n
        cy = height // 2
        r = 14

        for i, title in enumerate(self.STEP_TITLES):
            cx = int(step_w * i + step_w / 2)

            # Draw connector line to next step
            if i < n - 1:
                next_cx = int(step_w * (i + 1) + step_w / 2)
                line_y = cy
                # Line color: green if both steps completed, gray otherwise
                if i in self._completed and (i + 1) in self._completed:
                    color = QColor("#27ae60")
                elif i in self._completed:
                    color = QColor("#0078d7")
                else:
                    color = QColor("#cccccc")
                painter.setPen(color)
                painter.drawLine(cx + r, line_y, next_cx - r, line_y)

            # Determine colors
            if i in self._errors:
                bg_color = QColor("#e74c3c")
                text_color = QColor("#ffffff")
            elif i == self._current:
                bg_color = QColor("#0078d7")
                text_color = QColor("#ffffff")
            elif i in self._completed:
                bg_color = QColor("#27ae60")
                text_color = QColor("#ffffff")
            else:
                bg_color = QColor("#f0f0f0")
                text_color = QColor("#666666")

            # Draw circle
            painter.setBrush(bg_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

            # Draw number or checkmark
            painter.setPen(text_color)
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)

            if i in self._completed and i != self._current:
                text = "✓"
            else:
                text = str(i + 1)
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            painter.drawText(cx - tw // 2, cy + th // 2 - 3, text)

            # Draw title below
            painter.setPen(QColor("#333333") if i == self._current or i in self._errors else QColor("#888888"))
            font2 = QFont()
            font2.setPointSize(8)
            painter.setFont(font2)
            fm2 = QFontMetrics(font2)
            tw2 = fm2.horizontalAdvance(title)
            painter.drawText(cx - tw2 // 2, cy + r + 14, title)

        painter.end()
