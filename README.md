# CAB-F 工具

CAB-F 缝纫点与连边标注、数据处理、训练数据导出工具集。

当前版本以 GUI 工作台为主入口，适合标注、质检、裁剪和流程编排场景。

## 环境要求

- Python 3.10+
- 依赖安装：`pip install -r requirements.txt`

## 启动

```bash
# GUI 工作台
python main.py

# 流程命令行
cabf-flow --help
```

## 子功能

### 缝纫点与连边标注数据处理

8 步流程：配置检查 → 缝纫点预测 → 缝纫点修正 → 连边预测 → 连边修正 → 数据校验 → 数据导出 → 模型训练

### 关键字划分

根据文件名中的关键字将图片自动分类到子文件夹。默认关键字为 `top` 和 `bottom`，也支持自定义关键字。

### 批量裁剪

在首张图片上框选多个矩形 ROI，自动对所有图片进行批量裁剪。不同尺寸的图片会按比例缩放 ROI 坐标。

### 自动裁剪

将图片按固定尺寸（默认 `256 x 256`）自动切割成多个小块。支持两种边缘策略：

- 开启“允许重叠补边”时，边缘切块会回退补齐，保持输出尺寸一致。
- 关闭该选项时，边缘切块保留原始剩余尺寸。

## 命令行能力

```bash
cabf-flow doctor                    # 环境体检
cabf-flow show-config               # 查看当前配置
cabf-flow validate                  # 校验母标注
cabf-flow export                    # 导出模型 A/B 训练集
cabf-flow pipeline --dry-run        # 一键全流程 dry-run
cabf-flow settings                  # 打开 GUI 配置编辑器
```

说明：当前仓库主入口是 `main.py` 与 `cabf-flow` 脚本，不再使用旧版 `cli.py`。

## 项目结构

```
├── main.py                          # GUI 入口
├── pyproject.toml                   # 包配置
├── core/                            # 核心处理逻辑
│   ├── image_io.py                  # 图像读写
│   ├── cabf_shared.py               # CAB-F 共享层桥接
│   ├── cabf_dataset.py              # CAB-F 数据集兼容层
│   ├── keyword_split.py             # 关键字划分
│   ├── batch_crop.py                # 批量裁剪
│   └── auto_tile_crop.py            # 自动裁剪
├── gui/                             # PySide6 图形界面
│   ├── main_window.py               # 主窗口（工具导航中心）
│   ├── tools/                       # 工具页面
│   │   ├── base.py                  # 工具页面基类
│   │   ├── stitch_workflow.py       # 缝纫点与连边流程
│   │   ├── keyword_split.py         # 关键字划分
│   │   ├── batch_crop.py            # 批量裁剪
│   │   └── auto_tile_crop.py        # 自动裁剪
│   ├── cabf_dataset_tool.py         # 数据集校验与导出
│   ├── stitch_graph_editor.py       # 连边标注编辑器
│   ├── stitch_point_editor.py       # 缝纫点编辑器
│   ├── stitch_point_filter.py       # 数据筛选工具
│   ├── preview_widget.py            # 图像预览组件
│   └── project_tools/               # 项目工具注册
├── project_modules/
│   └── cabf_pipeline/               # CAB-F 流程编排
└── tests/                           # 单元测试
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
