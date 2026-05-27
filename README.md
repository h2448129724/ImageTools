# 图像处理工具箱

面向深度学习的图像处理和数据集准备工具，提供可视化 GUI 操作界面 + CLI 命令行。

## 环境要求

- Python 3.10+
- 依赖安装：`pip install -r requirements.txt`

## 启动

```bash
# GUI 模式
python main.py

# CLI 模式
imagetools --help
```

## CAB-F 数据流程

CAB-F 相关的数据生产、校验、导出规则已经单独固化到下面两份文档：

- [docs/CABF_DATASET_SOP.md](docs/CABF_DATASET_SOP.md)
- [docs/CABF_MASTER_SCHEMA.md](docs/CABF_MASTER_SCHEMA.md)

其中：

- `CABF_DATASET_SOP.md` 规定了旧数据迁移、新数据入库、模型预标 + 人工修正的标准流程
- `CABF_MASTER_SCHEMA.md` 固定了 `points + edges + metadata` 母格式以及当前版本 `1.2`

GUI 入口：

- `工具 -> CAB-F`

### CAB-F 项目工具中心

GUI 中的项目型工具已经整理为统一入口：

- `工具 -> CAB-F` 会先进入 CAB-F 工具中心
- 在弹窗中切换具体功能，而不是把所有 CAB-F 工具直接平铺在主菜单里

当前 CAB-F 工具中心已接入：

- 连边标注器
- 缝纫点数据筛选
- 数据集校验与导出

同时，项目工具入口已经改成“模块自注册”结构，后续新增 `VGS` 之类的项目时，只需要在 `gui/project_tools/` 下新增一个项目模块，主菜单会自动发现并显示。

### CAB-F 数据校验与导出

CAB-F 数据集校验与导出流程已经更新：

- 校验结果除了汇总计数外，还会输出有问题的样本名
- 问题会按样本合并展示，便于直接排查
- 导出时只需要选择一个导出根目录，程序会自动创建 `images`、`annotations`、`error`
- 如果样本标签存在问题，对应图片和标注不会进入正常导出集，而是统一进入 `error`

### CAB-F 标注器自动保存

CAB-F 两个编辑器已经统一接入自动保存状态机制：

- 连边标注器
- 缝纫点编辑器

保存状态会以统一的状态灯显示，包括：

- `未修改`
- `未保存修改`
- `自动保存中`
- `已自动保存`
- `保存失败`

当已启用覆盖原标签或已设置输出路径时，编辑过程中的修改会自动静默保存。

## CLI 命令

```bash
imagetools split <input_dir> <output_dir>          # 随机划分数据集
imagetools stratified-split <input_dir> <out>       # 分层划分
imagetools kfold <input_dir> <output_dir> --k 5     # K 折交叉验证
imagetools dedup <input_dir> [--mode perceptual]    # 图片去重（精确/感知）
imagetools tile <input> <output_dir>                # 大图切块
imagetools rename <input_dir> <output_dir>          # 批量重命名
imagetools resize <input_dir> <output_dir>          # 批量缩放
imagetools convert <input_dir> <output_dir>         # 批量格式转换
imagetools border <input_dir> <output_dir>          # 批量添加边框
imagetools augment <input_dir> <output_dir>         # 数据增强
imagetools format yolo2coco <img_dir> <ann_dir>     # 标注格式转换
imagetools annot validate <ann_dir> <img_dir>       # YOLO 标注校验
imagetools annot stats <ann_dir> <img_dir>          # 标注统计
```

## 功能概览

### 颜色转换
BGR ↔ RGB ↔ HSV ↔ LAB ↔ YUV ↔ Gray ↔ HLS ↔ YCrCb 等 11 种颜色空间互转。

### 基础图像处理
| 功能 | 说明 |
|------|------|
| 缩放 | 按宽高或比例缩放，支持保持宽高比 |
| 裁剪 | 指定 ROI 坐标和尺寸裁剪 |
| 中心裁剪 | 以图像中心为基准裁剪 |
| 旋转 | 任意角度旋转 |
| 翻转 | 水平 / 垂直 / 双向翻转 |
| 亮度/对比度 | 像素级调整 |
| 饱和度调整 | HSV 空间饱和度变换 |
| 直方图均衡化 | 全局均衡化 / CLAHE 自适应均衡化 |
| 二值化/阈值 | Otsu / 固定阈值 / 自适应均值 / 自适应高斯 |
| 形态学操作 | 腐蚀 / 膨胀 / 开运算 / 闭运算 |
| 填充/边框 | 常数填充 / 反射填充 / 复制填充 |
| 移除/添加 Alpha 通道 | 透明背景转白底等 |
| 图像叠加 | 贴图、加水印，支持透明度 |
| 通道提取 | 提取单个 B/G/R/A 通道 |
| 格式转换 | PNG/JPG/BMP/TIFF/WebP 互转 |

### 图像滤波
| 功能 | 说明 |
|------|------|
| 均值模糊 / 高斯模糊 | 平滑去噪 |
| 中值滤波 | 椒盐噪声去除 |
| 双边滤波 | 保边去噪 |
| 锐化 | 拉普拉斯锐化核 |
| Canny / Sobel / Laplacian | 边缘检测 |

### 大图切块
| 功能 | 说明 |
|------|------|
| 固定尺寸切块 | 将大图切成固定大小（如 256×256）的块，支持重叠 |
| 网格切块 | 按行列数均匀切分 |
| 分割标注切块 | 图片 + Labelme JSON 多边形标注联合切块，自动裁剪多边形并转换坐标系 |

### 数据增强
基于可组合 Pipeline 的数据增强框架，支持 17 种变换：

| 类别 | 变换 |
|------|------|
| 几何变换 | RandomHorizontalFlip, RandomVerticalFlip, RandomRotate, RandomScale, RandomCrop, LetterboxResize, LongestMaxSize |
| 颜色变换 | ColorJitter（亮度/对比度/饱和度/色相） |
| 噪声 | GaussianNoise, SaltAndPepperNoise |
| 模糊 | RandomGaussianBlur, RandomMotionBlur |
| 遮挡 | RandomErasing, Cutout |
| 归一化 | Normalize（ImageNet 均值/标准差） |
| 多图融合 | MixUp, Mosaic |

支持 Pipeline 配置的 JSON 序列化/反序列化，以及 `--config` 文件驱动的 CLI 批量增强。

#### BBox-Aware 增强管道
图像 + YOLO 边界框联合变换，适用于目标检测数据增强：
- BBoxHorizontalFlip, BBoxVerticalFlip
- BBoxScale, BBoxLetterboxResize, BBoxRandomCrop
- BBoxColorJitter（仅变换图像，框不变）

### 数据集处理
| 功能 | 说明 |
|------|------|
| 随机划分 | 按比例随机分 train/val/test |
| 分层划分 | 按子文件夹（类别）分层采样 |
| K 折交叉验证 | 生成 K 个 fold |
| YOLO ↔ COCO ↔ VOC | 标注格式互转 |
| 分类数据集生成 | 文件夹→ImageNet 风格数据集 |

### 标注工具
| 功能 | 说明 |
|------|------|
| YOLO / COCO 标注可视化 | 在图上绘制 bounding box |
| YOLO 标注校验 | 检测越界框、零面积框 |
| 标注统计 | 类别分布、框尺寸分布 |
| ROI 裁剪 | 根据标注框自动裁剪目标区域 |
| Mask ↔ 多边形 | 二值 Mask 与 Labelme JSON 多边形互转 |
| 标注增强 | YOLO / Labelme 标注联合增强（翻转/旋转/缩放/裁剪） |

### 批量处理
| 功能 | 说明 |
|------|------|
| 批量重命名 | 按规则统一重命名 |
| 批量缩放 | 统一调整图片尺寸（支持按比例） |
| 批量 ROI 裁剪 | 所有图片裁同一位置区域 |
| 批量格式转换 | PNG/JPG/BMP/TIFF/WebP 互转 |
| 批量添加边框 | 统一加边框 |
| 图片去重 | 精确切重（MD5）+ 感知去重（dHash 汉明距离） |

## 操作方式

1. **选择输入**：点击"选择文件"或"选择文件夹"，支持拖拽
2. **选择功能**：在功能树中点击目标功能，可用搜索框过滤
3. **调整参数**：右侧参数面板自动显示对应参数
4. **预览（推荐）**：点击"预览"按钮，在右侧查看单张图片的处理效果，不保存
5. **设置输出目录**：选择处理后文件的保存位置
6. **执行处理**：点击"执行处理"开始批量/单张处理

补充说明：

- 左侧参数区现在主要用于参数设置
- `预览`、`保存当前结果`、`执行处理` 已整理到主窗口底部操作区
- 预览区的坐标拾取、多边形绘制、ROI 框选已整理为统一的“采集工具”面板

## 快捷键

| 按键 | 功能 |
|------|------|
| Ctrl+O | 打开图片 |
| Ctrl+D | 打开文件夹 |
| Ctrl+R / F5 | 执行处理 |
| Ctrl+Z | 撤销 |
| Ctrl+Shift+Z / Ctrl+Y | 重做 |
| Ctrl+Q | 退出 |
| A | 上一张图片 |
| D | 下一张图片 |
| Esc | 退出坐标拾取模式 |

## 特性

- **EXIF 自动旋转**：读取 JPEG 时自动根据 EXIF 方向标签旋转图片
- **Unicode 路径支持**：通过 imencode/imdecode 支持中文/特殊字符文件名
- **并行处理**：批量操作使用 ThreadPoolExecutor 并行加速
- **Undo/Redo**：最多 20 步撤销历史
- **窗口状态持久化**：自动记住窗口大小和上次使用的功能
- **安全 XML 生成**：使用 lxml etree 防止注入
- **Header-only 图片信息**：PNG/JPEG 无需完整解码即可获取尺寸

## 项目结构

```
ImageTools/
├── main.py                     # GUI 入口
├── cli.py                      # CLI 入口
├── pyproject.toml              # 包配置 + 入口点
├── requirements.txt            # 依赖
├── core/                       # 核心处理逻辑
│   ├── image_io.py             # 图像读写 + EXIF 自动旋转
│   ├── color_conversion.py     # 颜色空间转换
│   ├── basic_processing.py     # 基础处理、滤波、形态学
│   ├── augmentation.py         # 可组合数据增强 Pipeline
│   ├── tiling.py               # 大图切块
│   ├── segmentation_tiling.py  # 分割标注联合切块
│   ├── dataset_split.py        # 数据集划分
│   ├── format_conversion.py    # 标注格式转换
│   ├── batch_processing.py     # 批量操作 + 感知去重
│   ├── annotation.py           # 标注可视化/校验/统计
│   ├── annotation_augment.py   # 标注联合增强
│   └── mask_polygon.py         # Mask ↔ 多边形转换
├── gui/                        # PySide6 图形界面
│   ├── main_window.py          # 主窗口 + Undo/Redo
│   ├── input_panel.py          # 输入面板
│   ├── function_panel.py       # 功能树 + 搜索
│   ├── param_panel.py          # 动态参数面板
│   ├── preview_widget.py       # 图像预览 + 坐标拾取
│   ├── output_panel.py         # 输出设置
│   ├── autosave.py             # 编辑器自动保存状态控制
│   ├── project_tools_hub.py    # 项目工具中心弹窗
│   ├── project_tools_registry.py # 项目工具自动发现
│   ├── project_tools/          # 项目模块自注册目录
│   └── workers.py              # 后台线程
├── utils/
│   └── helpers.py              # 文件工具、哈希
└── tests/                      # 289 个单元测试
    ├── test_augmentation.py
    ├── test_annotation_augment.py
    ├── test_annotation.py
    ├── test_batch_processing.py
    ├── test_basic_processing.py
    ├── test_color_conversion.py
    ├── test_dataset_split.py
    ├── test_format_conversion.py
    ├── test_helpers.py
    ├── test_image_io.py
    ├── test_mask_polygon.py
    ├── test_segmentation_tiling.py
    └── test_tiling.py
```

## 测试

```bash
python -m pytest tests/ -v
```

## 依赖

- PySide6 — GUI 框架
- OpenCV (cv2) — 图像处理
- NumPy — 数值计算
- lxml — XML 安全生成
