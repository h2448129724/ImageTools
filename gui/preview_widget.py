"""Image preview widget with zoom, pan, and pixel coordinate picker."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QScrollArea, QPushButton, QSlider, QTabWidget, QApplication)
from PySide6.QtCore import Qt, QSize, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QPainter, QPen, QColor
import cv2
import numpy as np


def cv2_to_qpixmap(img):
    """Convert OpenCV image (BGR) to QPixmap for display."""
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    elif img.shape[2] == 4:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgb.data, w, h, w * 4, QImage.Format_RGBA8888)
    else:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ZoomableLabel(QWidget):
    """Label that supports zoom via scroll wheel, pan via drag, and pixel coordinate picking."""
    zoomChanged = Signal(float)
    pixelClicked = Signal(int, int)   # x, y in image pixel coordinates
    pixelMoved = Signal(int, int)     # live tracking in picker mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._dragging = False
        self._drag_start = None
        self._base_offset = (0, 0)
        self._picker_mode = False
        self._marker_point = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    @property
    def picker_mode(self):
        return self._picker_mode

    @picker_mode.setter
    def picker_mode(self, value):
        self._picker_mode = value
        if value:
            self.setCursor(Qt.CrossCursor)
            self._marker_point = None
        else:
            self.setCursor(Qt.ArrowCursor)
            self._marker_point = None
        self.update()

    def _screen_to_image(self, sx, sy):
        """Convert screen coordinates to image pixel coordinates."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self.width(), self.height()
        scaled_w = self._pixmap.width() * self._zoom
        scaled_h = self._pixmap.height() * self._zoom
        img_x = int((sx - (pw - scaled_w) / 2 - self._offset_x) / self._zoom)
        img_y = int((sy - (ph - scaled_h) / 2 - self._offset_y) / self._zoom)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if 0 <= img_x < iw and 0 <= img_y < ih:
            return img_x, img_y
        return None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._marker_point = None
        self.fit_to_view()
        self.update()

    def fit_to_view(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw, ph = self.width() - 20, self.height() - 20
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0:
            return
        self._zoom = min(pw / iw, ph / ih, 1.0)
        self._offset_x = 0
        self._offset_y = 0
        self.zoomChanged.emit(self._zoom)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        pw = self.width()
        ph = self.height()
        scaled_w = self._pixmap.width() * self._zoom
        scaled_h = self._pixmap.height() * self._zoom
        x = int((pw - scaled_w) // 2 + self._offset_x)
        y = int((ph - scaled_h) // 2 + self._offset_y)
        target = QRectF(x, y, scaled_w, scaled_h)
        source = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target, self._pixmap, source)

        # Draw crosshair marker in picker mode
        if self._picker_mode and self._marker_point is not None:
            mx, my = self._marker_point
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            # Horizontal line
            painter.drawLine(QPointF(x, my), QPointF(x + scaled_w, my))
            # Vertical line
            painter.drawLine(QPointF(mx, y), QPointF(mx, y + scaled_h))
            # Center dot
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawPoint(QPointF(mx, my))

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = max(0.01, min(10.0, self._zoom * factor))
        self._zoom = new_zoom
        self.zoomChanged.emit(self._zoom)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._picker_mode:
                # Get image pixel coordinates
                pos = event.position()
                img_coord = self._screen_to_image(pos.x(), pos.y())
                if img_coord:
                    self._marker_point = (pos.x(), pos.y())
                    self.pixelClicked.emit(img_coord[0], img_coord[1])
                    self.update()
            else:
                self._dragging = True
                self._drag_start = event.position().toPoint()
                self._base_offset = (self._offset_x, self._offset_y)
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            pos = event.position().toPoint()
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            self._offset_x = self._base_offset[0] + dx
            self._offset_y = self._base_offset[1] + dy
            self.update()
        elif self._picker_mode:
            pos = event.position()
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                self.pixelMoved.emit(img_coord[0], img_coord[1])

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False
        if not self._picker_mode:
            self.setCursor(Qt.ArrowCursor)


class PreviewPanel(QWidget):
    """Preview panel with before/after tabs and coordinate picker."""
    imageDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._picker_active = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self._on_fit)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        self.zoom_label = QLabel("100%")

        # Picker toggle button
        self.btn_picker = QPushButton("坐标拾取")
        self.btn_picker.setObjectName("btnPicker")
        self.btn_picker.setCheckable(True)
        self.btn_picker.toggled.connect(self._on_picker_toggled)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #c0392b; font-weight: bold; min-width: 160px;")

        toolbar.addWidget(self.btn_fit)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.zoom_label)
        toolbar.addSpacing(16)
        toolbar.addWidget(self.btn_picker)
        toolbar.addWidget(self.coord_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Tabs for before/after
        self.tabs = QTabWidget()
        self.original_view = ZoomableLabel()
        self.result_view = ZoomableLabel()
        self.tabs.addTab(self.original_view, "原图")
        self.tabs.addTab(self.result_view, "处理后")
        layout.addWidget(self.tabs)

        self.current_image = None

        # Connect picker signals
        self.original_view.pixelClicked.connect(self._on_pixel_clicked)
        self.original_view.pixelMoved.connect(self._on_pixel_moved)
        self.result_view.pixelClicked.connect(self._on_pixel_clicked)
        self.result_view.pixelMoved.connect(self._on_pixel_moved)

    def set_original(self, img):
        """Set the original image for preview."""
        self.current_image = img
        self.original_view.set_pixmap(cv2_to_qpixmap(img))

    def set_result(self, img):
        """Set the result image and switch to result tab."""
        self.result_view.set_pixmap(cv2_to_qpixmap(img))
        self.tabs.setCurrentIndex(1)

    def _on_picker_toggled(self, checked):
        self._picker_active = checked
        self.original_view.picker_mode = checked
        self.result_view.picker_mode = checked
        if not checked:
            self.coord_label.setText("")

    def _on_pixel_clicked(self, x, y):
        coord_text = f"X={x}  Y={y}"
        self.coord_label.setText(coord_text)
        # Copy to clipboard
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(f"({x}, {y})")

    def _on_pixel_moved(self, x, y):
        if self._picker_active:
            self.coord_label.setText(f"X={x}  Y={y}")

    def _on_fit(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current.fit_to_view()
            self._update_zoom_label()

    def _on_zoom_in(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current._zoom *= 1.25
            current.update()
            self._update_zoom_label()

    def _on_zoom_out(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current._zoom *= 0.8
            current.update()
            self._update_zoom_label()

    def _update_zoom_label(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            self.zoom_label.setText(f"{int(current._zoom * 100)}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.original_view.fit_to_view()
        self.result_view.fit_to_view()
