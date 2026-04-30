"""Function selection panel with search and categorized tree."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                                QLabel, QLineEdit, QHBoxLayout)
from PySide6.QtCore import Signal, Qt


FUNCTION_REGISTRY = {
    "颜色转换": [
        ("color_bgr2rgb", "BGR → RGB"), ("color_rgb2bgr", "RGB → BGR"),
        ("color_bgr2hsv", "BGR → HSV"), ("color_hsv2bgr", "HSV → BGR"),
        ("color_bgr2lab", "BGR → LAB"), ("color_lab2bgr", "LAB → BGR"),
        ("color_bgr2gray", "BGR → 灰度"), ("color_gray2bgr", "灰度 → BGR"),
        ("color_bgr2yuv", "BGR → YUV"), ("color_bgr2hls", "BGR → HLS"),
        ("color_bgr2ycrcb", "BGR → YCrCb"),
    ],
    "基础处理": [
        ("resize", "缩放"), ("crop", "裁剪"), ("center_crop", "中心裁剪"),
        ("rotate", "旋转"), ("flip", "翻转"),
        ("brightness_contrast", "亮度/对比度"), ("saturation", "饱和度调整"),
        ("histogram_eq", "直方图均衡化"), ("threshold", "二值化/阈值"),
        ("morphology", "形态学操作"), ("pad", "填充/边框"),
        ("remove_alpha", "移除Alpha通道"), ("add_alpha", "添加Alpha通道"),
        ("overlay", "图像叠加"), ("channel_extract", "通道提取"),
    ],
    "图像滤波": [
        ("filter_blur", "均值模糊"), ("filter_gaussian", "高斯模糊"),
        ("filter_median", "中值滤波"), ("filter_bilateral", "双边滤波"),
        ("filter_sharpen", "锐化"),
        ("edge_canny", "Canny边缘检测"), ("edge_sobel", "Sobel边缘检测"),
        ("edge_laplacian", "Laplacian边缘检测"),
    ],
    "大图切块": [
        ("tile_fixed", "固定尺寸切块"), ("tile_grid", "网格切块"),
        ("seg_tile", "分割标注切块(图+标签)"),
    ],
    "数据集处理": [
        ("dataset_random_split", "随机划分"), ("dataset_stratified_split", "分层划分"),
        ("dataset_kfold", "K折交叉验证"),
        ("format_yolo2coco", "YOLO → COCO"), ("format_coco2yolo", "COCO → YOLO"),
        ("format_voc2yolo", "VOC → YOLO"), ("format_voc2coco", "VOC → COCO"),
        ("format_coco2voc", "COCO → VOC"), ("format_classification", "生成分类数据集"),
    ],
    "标注工具": [
        ("annot_draw_yolo", "YOLO标注可视化"), ("annot_draw_coco", "COCO标注可视化"),
        ("annot_validate_yolo", "YOLO标注校验"), ("annot_statistics", "标注统计"),
        ("annot_crop_roi", "标注ROI裁剪"),
    ],
    "批量处理": [
        ("batch_rename", "批量重命名"), ("batch_resize", "批量缩放"),
        ("batch_roi_crop", "批量ROI裁剪"), ("batch_convert_format", "批量格式转换"),
        ("batch_add_border", "批量添加边框"), ("batch_deduplicate", "图片去重"),
    ],
}


class FunctionPanel(QWidget):
    functionSelected = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_items = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("功能选择")
        f = title.font(); f.setBold(True); title.setFont(f)
        layout.addWidget(title)

        # Search box
        self.search = QLineEdit()
        self.search.setObjectName("searchBox")
        self.search.setPlaceholderText("搜索功能...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.itemClicked.connect(self._on_item_clicked)

        for cat_name, functions in FUNCTION_REGISTRY.items():
            cat = QTreeWidgetItem([cat_name])
            cat.setFlags(cat.flags() & ~Qt.ItemIsSelectable)
            cat.setData(0, Qt.UserRole, "__category__")
            font = cat.font(0)
            font.setBold(True)
            cat.setFont(0, font)
            for key, name in functions:
                item = QTreeWidgetItem([name])
                item.setData(0, Qt.UserRole, key)
                item.setData(0, Qt.UserRole + 1, name)
                cat.addChild(item)
                self._all_items.append((item, name, cat_name))
            self.tree.addTopLevelItem(cat)

        self.tree.expandAll()
        layout.addWidget(self.tree)

    def _on_search(self, text):
        """Filter tree by search text while preserving category visibility."""
        if not text.strip():
            for i in range(self.tree.topLevelItemCount()):
                cat = self.tree.topLevelItem(i)
                cat.setHidden(False)
                for j in range(cat.childCount()):
                    cat.child(j).setHidden(False)
            return

        low = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(cat.childCount()):
                child = cat.child(j)
                name = child.text(0).lower()
                visible = low in name
                child.setHidden(not visible)
                if visible:
                    any_visible = True
            cat.setHidden(not any_visible)
            if any_visible and not cat.isExpanded():
                cat.setExpanded(True)

    def _on_item_clicked(self, item, _):
        key = item.data(0, Qt.UserRole)
        if key == "__category__":
            return
        name = item.data(0, Qt.UserRole + 1)
        if key:
            self.functionSelected.emit(key, name)
