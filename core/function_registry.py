"""Unified function registry: single source of truth for all function definitions.

Replaces the separate FUNCTION_REGISTRY (function_panel.py), PARAM_SPECS (param_panel.py),
and _build_handlers (main_window.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionDef:
    category: str
    key: str
    name: str
    params: list[dict[str, Any]] = field(default_factory=list)
    handler_type: str = "simple"  # "simple" | "complex" | "batch"
    single_image: bool = False   # True if function can work on a single loaded image
    description: str = ""


FUNCTION_DEFS: list[FunctionDef] = [

    # ==================== 颜色转换 ====================
    *[FunctionDef("颜色转换", k, n) for k, n in [
        ("color_bgr2rgb", "BGR → RGB"), ("color_rgb2bgr", "RGB → BGR"),
        ("color_bgr2hsv", "BGR → HSV"), ("color_hsv2bgr", "HSV → BGR"),
        ("color_bgr2lab", "BGR → LAB"), ("color_lab2bgr", "LAB → BGR"),
        ("color_bgr2gray", "BGR → 灰度"), ("color_gray2bgr", "灰度 → BGR"),
        ("color_bgr2yuv", "BGR → YUV"), ("color_yuv2bgr", "YUV → BGR"),
        ("color_bgr2hls", "BGR → HLS"), ("color_hls2bgr", "HLS → BGR"),
        ("color_bgr2ycrcb", "BGR → YCrCb"), ("color_ycrcb2bgr", "YCrCb → BGR"),
    ]],

    # ==================== 基础处理 ====================
    FunctionDef("基础处理", "resize", "缩放", [
        {"name": "width", "label": "宽度", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "height", "label": "高度", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "keep_aspect", "label": "保持比例", "type": "bool", "default": True},
        {"name": "scale", "label": "缩放比例(0=使用宽高)", "type": "float", "default": 0, "min": 0, "max": 10, "step": 0.01},
    ]),
    FunctionDef("基础处理", "crop", "裁剪", [
        {"name": "x", "label": "X坐标", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "y", "label": "Y坐标", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "w", "label": "宽度", "type": "int", "default": 256, "min": 1, "max": 10000},
        {"name": "h", "label": "高度", "type": "int", "default": 256, "min": 1, "max": 10000},
    ]),
    FunctionDef("基础处理", "center_crop", "中心裁剪", [
        {"name": "w", "label": "宽度", "type": "int", "default": 512, "min": 1, "max": 10000},
        {"name": "h", "label": "高度", "type": "int", "default": 512, "min": 1, "max": 10000},
    ]),
    FunctionDef("基础处理", "rotate", "旋转", [
        {"name": "angle", "label": "旋转角度", "type": "int", "default": 90, "min": -360, "max": 360},
        {"name": "keep_size", "label": "保持原尺寸", "type": "bool", "default": False},
    ]),
    FunctionDef("基础处理", "flip", "翻转", [
        {"name": "direction", "label": "翻转方向", "type": "combo",
         "options": ["horizontal", "vertical", "both"], "default": "horizontal"},
    ]),
    FunctionDef("基础处理", "brightness_contrast", "亮度/对比度", [
        {"name": "brightness", "label": "亮度(-255~255)", "type": "int", "default": 0, "min": -255, "max": 255},
        {"name": "contrast", "label": "对比度", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
    ]),
    FunctionDef("基础处理", "saturation", "饱和度调整", [
        {"name": "factor", "label": "饱和度因子", "type": "float", "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1},
    ]),
    FunctionDef("基础处理", "histogram_eq", "直方图均衡化", [
        {"name": "adaptive", "label": "自适应(CLAHE)", "type": "bool", "default": False},
        {"name": "clip_limit", "label": "对比度阈值", "type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5},
        {"name": "tile_size", "label": "网格大小", "type": "int", "default": 8, "min": 2, "max": 32},
    ]),
    FunctionDef("基础处理", "threshold", "二值化/阈值", [
        {"name": "method", "label": "方法", "type": "combo",
         "options": ["otsu", "binary", "adaptive_mean", "adaptive_gaussian"], "default": "otsu"},
        {"name": "thresh", "label": "阈值", "type": "int", "default": 127, "min": 0, "max": 255},
        {"name": "maxval", "label": "最大值", "type": "int", "default": 255, "min": 0, "max": 255},
        {"name": "block_size", "label": "邻域大小(自适应)", "type": "int", "default": 11, "min": 3, "max": 99},
    ]),
    FunctionDef("基础处理", "morphology", "形态学操作", [
        {"name": "op_type", "label": "操作", "type": "combo",
         "options": ["erode", "dilate", "open", "close"], "default": "erode"},
        {"name": "ksize", "label": "核大小", "type": "int", "default": 3, "min": 1, "max": 31},
        {"name": "iterations", "label": "迭代次数", "type": "int", "default": 1, "min": 1, "max": 10},
    ]),
    FunctionDef("基础处理", "pad", "填充/边框", [
        {"name": "top", "label": "上边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "bottom", "label": "下边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "left", "label": "左边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "right", "label": "右边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "mode", "label": "填充模式", "type": "combo",
         "options": ["constant", "reflect", "replicate"], "default": "constant"},
    ]),
    FunctionDef("基础处理", "remove_alpha", "移除Alpha通道", [
        {"name": "bg_color", "label": "背景色", "type": "combo",
         "options": ["white", "black"], "default": "white"},
    ]),
    FunctionDef("基础处理", "add_alpha", "添加Alpha通道"),
    FunctionDef("基础处理", "overlay", "图像叠加", [
        {"name": "overlay_path", "label": "叠加图片", "type": "file", "default": ""},
        {"name": "x", "label": "X坐标", "type": "int", "default": 0, "min": -5000, "max": 10000},
        {"name": "y", "label": "Y坐标", "type": "int", "default": 0, "min": -5000, "max": 10000},
        {"name": "opacity", "label": "透明度", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    ]),
    FunctionDef("基础处理", "channel_extract", "通道提取", [
        {"name": "channel", "label": "通道", "type": "combo",
         "options": ["B (0)", "G (1)", "R (2)", "A (3)"], "default": "B (0)"},
    ]),
    FunctionDef("基础处理", "format_convert", "格式转换", [
        {"name": "fmt", "label": "目标格式", "type": "combo",
         "options": ["png", "jpg", "bmp", "tiff", "webp"], "default": "png"},
        {"name": "quality", "label": "质量", "type": "int", "default": 95, "min": 1, "max": 100},
    ]),

    # ==================== 图像滤波 ====================
    FunctionDef("图像滤波", "filter_blur", "均值模糊", [
        {"name": "ksize", "label": "核大小", "type": "int", "default": 5, "min": 1, "max": 31},
    ]),
    FunctionDef("图像滤波", "filter_gaussian", "高斯模糊", [
        {"name": "ksize", "label": "核大小(奇数)", "type": "int", "default": 5, "min": 1, "max": 31},
    ]),
    FunctionDef("图像滤波", "filter_median", "中值滤波", [
        {"name": "ksize", "label": "核大小(奇数)", "type": "int", "default": 5, "min": 1, "max": 31},
    ]),
    FunctionDef("图像滤波", "filter_bilateral", "双边滤波", [
        {"name": "ksize", "label": "直径", "type": "int", "default": 9, "min": 1, "max": 31},
    ]),
    FunctionDef("图像滤波", "filter_sharpen", "锐化"),
    FunctionDef("图像滤波", "edge_canny", "Canny边缘检测", [
        {"name": "threshold1", "label": "低阈值", "type": "int", "default": 100, "min": 0, "max": 500},
        {"name": "threshold2", "label": "高阈值", "type": "int", "default": 200, "min": 0, "max": 500},
    ]),
    FunctionDef("图像滤波", "edge_sobel", "Sobel边缘检测"),
    FunctionDef("图像滤波", "edge_laplacian", "Laplacian边缘检测"),

    # ==================== 大图切块 ====================
    FunctionDef("大图切块", "tile_fixed", "固定尺寸切块", [
        {"name": "tile_w", "label": "切块宽度", "type": "int", "default": 512, "min": 32, "max": 4096},
        {"name": "tile_h", "label": "切块高度", "type": "int", "default": 512, "min": 32, "max": 4096},
        {"name": "overlap", "label": "重叠像素", "type": "int", "default": 0, "min": 0, "max": 2048},
        {"name": "discard_incomplete", "label": "丢弃不完整块", "type": "bool", "default": True},
    ], handler_type="complex", single_image=True),
    FunctionDef("大图切块", "tile_grid", "网格切块", [
        {"name": "rows", "label": "行数", "type": "int", "default": 3, "min": 1, "max": 100},
        {"name": "cols", "label": "列数", "type": "int", "default": 3, "min": 1, "max": 100},
    ], handler_type="complex", single_image=True),
    FunctionDef("大图切块", "seg_tile", "分割标注切块(图+标签)", [
        {"name": "tile_w", "label": "切块宽度", "type": "int", "default": 256, "min": 32, "max": 4096},
        {"name": "tile_h", "label": "切块高度", "type": "int", "default": 256, "min": 32, "max": 4096},
        {"name": "overlap", "label": "重叠像素", "type": "int", "default": 0, "min": 0, "max": 2048},
        {"name": "discard_empty", "label": "丢弃无标注块", "type": "bool", "default": False},
        {"name": "discard_incomplete", "label": "丢弃不完整块", "type": "bool", "default": True},
        {"name": "ann_dir", "label": "标注文件夹(JSON)", "type": "dir", "default": ""},
    ], handler_type="batch"),

    # ==================== 数据集处理 ====================
    FunctionDef("数据集处理", "dataset_random_split", "随机划分", [
        {"name": "train_ratio", "label": "训练集比例", "type": "float", "default": 0.7, "min": 0.1, "max": 0.9, "step": 0.05},
        {"name": "val_ratio", "label": "验证集比例", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05},
        {"name": "test_ratio", "label": "测试集比例", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
        {"name": "label_dir", "label": "标注文件夹(可选)", "type": "dir", "default": ""},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "dataset_stratified_split", "分层划分", [
        {"name": "train_ratio", "label": "训练集比例", "type": "float", "default": 0.7, "min": 0.1, "max": 0.9, "step": 0.05},
        {"name": "val_ratio", "label": "验证集比例", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05},
        {"name": "test_ratio", "label": "测试集比例", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "dataset_kfold", "K折交叉验证", [
        {"name": "k", "label": "折数", "type": "int", "default": 5, "min": 2, "max": 10},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_yolo2coco", "YOLO → COCO", [
        {"name": "yolo_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_coco2yolo", "COCO → YOLO", [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_voc2yolo", "VOC → YOLO", [
        {"name": "voc_dir", "label": "VOC标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_yolo2voc", "YOLO → VOC", [
        {"name": "yolo_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_voc2coco", "VOC → COCO", [
        {"name": "voc_dir", "label": "VOC标注目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_coco2voc", "COCO → VOC", [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_xanylabeling2yolo", "X-AnyLabeling → YOLO", [
        {"name": "src_dir", "label": "数据集目录", "type": "dir", "default": ""},
        {"name": "val_ratio", "label": "验证集比例", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
        {"name": "categories", "label": "类别名(逗号分隔,留空自动检测)", "type": "text", "default": ""},
    ], handler_type="batch"),
    FunctionDef("数据集处理", "format_classification", "图片/JSON分类", handler_type="batch"),

    # ==================== 标注工具 ====================
    FunctionDef("标注工具", "annot_draw_yolo", "YOLO标注可视化", [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1"},
    ], handler_type="complex", single_image=True),
    FunctionDef("标注工具", "annot_draw_coco", "COCO标注可视化", [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ], handler_type="complex", single_image=True),
    FunctionDef("标注工具", "annot_validate_yolo", "YOLO标注校验", [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "img_w", "label": "图片宽度", "type": "int", "default": 1920, "min": 1, "max": 100000},
        {"name": "img_h", "label": "图片高度", "type": "int", "default": 1080, "min": 1, "max": 100000},
    ], handler_type="batch"),
    FunctionDef("标注工具", "annot_statistics", "标注统计", [
        {"name": "ann_dir", "label": "标注目录", "type": "dir", "default": ""},
        {"name": "img_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "format_type", "label": "标注格式", "type": "combo", "options": ["yolo", "coco"], "default": "yolo"},
    ], handler_type="batch"),
    FunctionDef("标注工具", "annot_crop_roi", "标注ROI裁剪", [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1"},
        {"name": "padding", "label": "额外边距(px)", "type": "int", "default": 0, "min": 0, "max": 200},
    ], handler_type="complex", single_image=True),
    FunctionDef("标注工具", "mask_to_polygons", "Mask转多边形(Labelme)", [
        {"name": "mask_dir", "label": "Mask图片目录", "type": "dir", "default": ""},
        {"name": "label", "label": "标注标签名", "type": "text", "default": "object"},
        {"name": "epsilon_factor", "label": "简化精度", "type": "float",
         "default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001},
    ], handler_type="batch"),
    FunctionDef("标注工具", "polygons_to_mask", "多边形转Mask", [
        {"name": "ann_dir", "label": "Labelme JSON目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录(获取尺寸,可选)", "type": "dir", "default": ""},
        {"name": "default_h", "label": "默认高度", "type": "int", "default": 1080, "min": 1, "max": 10000},
        {"name": "default_w", "label": "默认宽度", "type": "int", "default": 1920, "min": 1, "max": 10000},
    ], handler_type="batch"),
    FunctionDef("标注工具", "augment_yolo", "标注增强(YOLO)", [
        {"name": "ann_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "transform_type", "label": "变换类型", "type": "combo",
         "options": ["flip", "rotate", "resize", "crop"], "default": "flip"},
        {"name": "direction", "label": "翻转方向(flip)", "type": "combo",
         "options": ["horizontal", "vertical", "both"], "default": "horizontal"},
        {"name": "angle", "label": "旋转角度(rotate)", "type": "int", "default": 90, "min": -360, "max": 360},
        {"name": "keep_size", "label": "保持原尺寸(rotate)", "type": "bool", "default": True},
        {"name": "new_w", "label": "目标宽度(resize)", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "new_h", "label": "目标高度(resize)", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "crop_x", "label": "裁剪X(crop)", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "crop_y", "label": "裁剪Y(crop)", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "crop_w", "label": "裁剪宽度(crop)", "type": "int", "default": 256, "min": 1, "max": 10000},
        {"name": "crop_h", "label": "裁剪高度(crop)", "type": "int", "default": 256, "min": 1, "max": 10000},
    ], handler_type="batch"),
    FunctionDef("标注工具", "augment_labelme", "标注增强(Labelme)", [
        {"name": "ann_dir", "label": "Labelme JSON目录", "type": "dir", "default": ""},
        {"name": "transform_type", "label": "变换类型", "type": "combo",
         "options": ["flip", "rotate", "resize", "crop"], "default": "flip"},
        {"name": "direction", "label": "翻转方向(flip)", "type": "combo",
         "options": ["horizontal", "vertical", "both"], "default": "horizontal"},
        {"name": "angle", "label": "旋转角度(rotate)", "type": "int", "default": 90, "min": -360, "max": 360},
        {"name": "keep_size", "label": "保持原尺寸(rotate)", "type": "bool", "default": True},
        {"name": "new_w", "label": "目标宽度(resize)", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "new_h", "label": "目标高度(resize)", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "crop_x", "label": "裁剪X(crop)", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "crop_y", "label": "裁剪Y(crop)", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "crop_w", "label": "裁剪宽度(crop)", "type": "int", "default": 256, "min": 1, "max": 10000},
        {"name": "crop_h", "label": "裁剪高度(crop)", "type": "int", "default": 256, "min": 1, "max": 10000},
    ], handler_type="batch"),
    FunctionDef("标注工具", "dataset_review", "数据集审查", handler_type="batch"),

    # ==================== 模型工具 ====================
    FunctionDef("模型工具", "export_onnx", "YOLO导出ONNX", [
        {"name": "model_path", "label": "YOLO模型文件(.pt)", "type": "file", "default": ""},
        {"name": "imgsz", "label": "输入图像尺寸", "type": "int", "default": 640, "min": 32, "max": 4096},
        {"name": "opset", "label": "ONNX Opset版本", "type": "int", "default": 11, "min": 9, "max": 20},
        {"name": "simplify", "label": "简化模型", "type": "bool", "default": True},
        {"name": "dynamic", "label": "动态输入尺寸", "type": "bool", "default": False},
        {"name": "half", "label": "FP16半精度", "type": "bool", "default": False},
        {"name": "device", "label": "设备(留空自动)", "type": "text", "default": ""},
    ], handler_type="batch"),
    FunctionDef("模型工具", "yolo_train", "YOLO一键训练", [
        {"name": "data", "label": "数据集YAML路径", "type": "file", "default": ""},
        {"name": "model", "label": "预训练模型", "type": "combo",
         "options": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                     "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
                     "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                     "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
         "default": "yolov8n.pt"},
        {"name": "epochs", "label": "训练轮数", "type": "int", "default": 100, "min": 1, "max": 10000},
        {"name": "imgsz", "label": "输入图像尺寸", "type": "int", "default": 640, "min": 32, "max": 4096},
        {"name": "batch", "label": "批次大小", "type": "int", "default": 16, "min": 1, "max": 256},
        {"name": "workers", "label": "数据加载线程数", "type": "int", "default": 8, "min": 0, "max": 32},
        {"name": "lr0", "label": "初始学习率", "type": "float", "default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001},
        {"name": "lrf", "label": "最终学习率因子", "type": "float", "default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001},
        {"name": "optimizer", "label": "优化器", "type": "combo",
         "options": ["SGD", "Adam", "AdamW", "Lion"], "default": "SGD"},
        {"name": "device", "label": "设备(留空自动)", "type": "text", "default": ""},
        {"name": "patience", "label": "早停耐心值", "type": "int", "default": 50, "min": 1, "max": 500},
        {"name": "seed", "label": "随机种子(0=随机)", "type": "int", "default": 0, "min": 0, "max": 99999},
        {"name": "project", "label": "输出项目目录", "type": "text", "default": "runs"},
        {"name": "name", "label": "训练名称", "type": "text", "default": "detect"},
        {"name": "augment", "label": "启用数据增强", "type": "bool", "default": False},
    ], handler_type="batch"),

    # ==================== 批量处理 ====================
    FunctionDef("批量处理", "batch_rename", "批量重命名", [
        {"name": "prefix", "label": "文件名前缀", "type": "text", "default": "img_"},
        {"name": "start_index", "label": "起始编号", "type": "int", "default": 1, "min": 0, "max": 99999},
        {"name": "digits", "label": "编号位数", "type": "int", "default": 4, "min": 1, "max": 10},
        {"name": "keep_ext", "label": "保持原格式", "type": "bool", "default": True},
    ], handler_type="batch"),
    FunctionDef("批量处理", "batch_resize", "批量缩放", [
        {"name": "width", "label": "目标宽度", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "height", "label": "目标高度", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "scale", "label": "缩放比例 (0=不使用)", "type": "float", "default": 0, "min": 0, "max": 10.0, "step": 0.1},
        {"name": "keep_aspect", "label": "保持比例", "type": "bool", "default": True},
    ], handler_type="batch"),
    FunctionDef("批量处理", "batch_roi_crop", "批量定点裁剪", [
        {"name": "x", "label": "起点X", "type": "int", "default": 0, "min": 0, "max": 99999},
        {"name": "y", "label": "起点Y", "type": "int", "default": 0, "min": 0, "max": 99999},
        {"name": "w", "label": "裁剪宽度", "type": "int", "default": 512, "min": 1, "max": 99999},
        {"name": "h", "label": "裁剪高度", "type": "int", "default": 512, "min": 1, "max": 99999},
    ], handler_type="complex", single_image=True),
    FunctionDef("批量处理", "batch_convert_format", "批量格式转换", [
        {"name": "fmt", "label": "目标格式", "type": "combo",
         "options": ["png", "jpg", "bmp", "tiff", "webp"], "default": "png"},
        {"name": "quality", "label": "质量", "type": "int", "default": 95, "min": 1, "max": 100},
    ], handler_type="batch"),
    FunctionDef("批量处理", "batch_add_border", "批量添加边框", [
        {"name": "border_size", "label": "边框大小", "type": "int", "default": 10, "min": 0, "max": 200},
        {"name": "color", "label": "颜色", "type": "combo",
         "options": ["black", "white", "red", "green", "blue"], "default": "black"},
    ], handler_type="complex", single_image=True),
    FunctionDef("批量处理", "batch_deduplicate", "图片去重", [
        {"name": "mode", "label": "去重模式", "type": "choice",
         "choices": ["exact", "perceptual"], "choice_labels": ["精确去重(完全相同)", "感知去重(视觉相似)"],
         "default": "exact"},
        {"name": "similarity_threshold", "label": "相似度阈值(感知模式)", "type": "int",
         "default": 10, "min": 0, "max": 64},
    ], handler_type="batch"),
]


# ---- Lookup helpers ----

_def_by_key: dict[str, FunctionDef] = {d.key: d for d in FUNCTION_DEFS}


def get_function_def(key: str) -> FunctionDef | None:
    d = _def_by_key.get(key)
    if d and not d.description:
        # Auto-generate a simple description from the name
        object.__setattr__(d, "description", f"{d.name} — 请在参数面板中配置相关选项")
    return d


def get_function_description(key: str) -> str:
    d = _def_by_key.get(key)
    if d and d.description:
        return d.description
    return ""


def can_single_image(key: str) -> bool:
    d = _def_by_key.get(key)
    return d.single_image if d else False


def get_categories() -> list[str]:
    seen: list[str] = []
    for d in FUNCTION_DEFS:
        if d.category not in seen:
            seen.append(d.category)
    return seen


def get_functions_by_category(category: str) -> list[tuple[str, str]]:
    return [(d.key, d.name) for d in FUNCTION_DEFS if d.category == category]


def get_param_specs(key: str) -> list[dict[str, Any]]:
    d = _def_by_key.get(key)
    return d.params if d else []


def get_all_functions_flat() -> list[tuple[str, str, str]]:
    """Return (key, name, category) for every function, for search."""
    return [(d.key, d.name, d.category) for d in FUNCTION_DEFS]


def get_function_registry_dict() -> dict[str, list[tuple[str, str]]]:
    """Build the category→[(key, name)] mapping (backward-compatible with old FUNCTION_REGISTRY)."""
    result: dict[str, list[tuple[str, str]]] = {}
    for d in FUNCTION_DEFS:
        result.setdefault(d.category, []).append((d.key, d.name))
    return result
