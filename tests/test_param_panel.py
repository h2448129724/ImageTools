"""Tests for gui.param_panel module."""
import pytest

try:
    from PySide6.QtWidgets import QApplication
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

pytestmark = pytest.mark.skipif(not _HAS_QT, reason="PySide6 not available")


@pytest.fixture(scope="module")
def app():
    """Create QApplication for tests."""
    existing = QApplication.instance()
    if existing:
        return existing
    return QApplication([])


@pytest.fixture
def panel(app):
    from gui.param_panel import ParamPanel
    return ParamPanel()


class TestParamPanel:
    def test_initial_state(self, panel):
        assert panel._current_key is None
        assert panel._widgets == {}

    def test_set_function_creates_widgets(self, panel):
        panel.set_function("resize", "缩放")
        assert panel._current_key == "resize"
        assert len(panel._widgets) > 0
        assert "width" in panel._widgets
        assert "height" in panel._widgets

    def test_set_function_clears_previous(self, panel):
        panel.set_function("resize", "缩放")
        assert "width" in panel._widgets
        panel.set_function("crop", "裁剪")
        assert "x" in panel._widgets
        assert "width" not in panel._widgets

    def test_collect_params_int(self, panel):
        panel.set_function("resize", "缩放")
        p = panel.get_params()["params"]
        assert "width" in p
        assert isinstance(p["width"], int)

    def test_collect_params_bool(self, panel):
        panel.set_function("resize", "缩放")
        p = panel.get_params()["params"]
        assert "keep_aspect" in p
        assert isinstance(p["keep_aspect"], bool)

    def test_collect_params_float(self, panel):
        panel.set_function("brightness_contrast", "亮度/对比度")
        p = panel.get_params()["params"]
        assert "contrast" in p
        assert isinstance(p["contrast"], float)

    def test_choice_widget(self, panel):
        panel.set_function("batch_deduplicate", "图片去重")
        assert "mode" in panel._widgets
        p = panel.get_params()["params"]
        assert p["mode"] in ("exact", "perceptual")

    def test_combo_widget(self, panel):
        panel.set_function("flip", "翻转")
        assert "direction" in panel._widgets
        p = panel.get_params()["params"]
        assert p["direction"] in ("horizontal", "vertical", "both")

    def test_no_params_function(self, panel):
        panel.set_function("filter_sharpen", "锐化")
        assert panel.get_params()["params"] == {}

    def test_title_updates(self, panel):
        panel.set_function("resize", "缩放")
        assert "缩放" in panel._title_label.text()
