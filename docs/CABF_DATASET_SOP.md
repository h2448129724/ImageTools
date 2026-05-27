# CAB-F 数据集 SOP

版本：`2026-05-25`

适用范围：

- CAB-F 缝纫点检测模型（模型 A）
- CAB-F 连边 / 缝制线路模型（模型 B）
- `D:\project\tianwei\img_tools` 中的 CAB-F 标注、校验、导出工具

---

## 1. 总体原则

CAB-F 数据只维护 **一套母标注**，格式统一为：

- `points`
- `edges`
- `metadata`

模型 A 和模型 B 的训练数据都从这套母标注自动导出，不再分别手工维护两套源数据。

---

## 2. 目录职责

建议长期固定为下面这类结构：

```text
CAB-F/
  master_annotations/
    images/
    annotations/
  model_a_export/
    images/
    annotations/
  model_b_export/
    images/
    annotations/
  predictions/
    points/
    edges/
```

说明：

- `master_annotations`：唯一真值源
- `model_a_export`：导出给点检测训练
- `model_b_export`：导出给连边训练
- `predictions`：模型自动结果，只作为中间产物，不能直接覆盖母标注

---

## 3. 旧数据迁移 SOP

### 3.1 旧的点标注数据

旧目录例如：

- `sew_point/annotations`

这类数据通常只有 LabelMe `point` 标注，没有边。

迁移流程：

1. 用 CAB-F 数据集工具读取旧点标注
2. 转成母格式
3. 得到：
   - `points`：来自原始点
   - `edges`：先为空
   - `metadata`：自动补齐来源信息
4. 保存到 `master_annotations/annotations`

这一步完成后，数据已经符合母格式，但还不能直接作为模型 B 真值，因为 `edges` 还没补。

### 3.2 已有边标注数据

旧目录例如：

- `sew_point_connect/annotations`

迁移流程：

1. 用 CAB-F 数据集工具读取
2. 执行校验
3. 统一字段
4. 规范化保存到 `master_annotations/annotations`

---

## 4. 新数据入库 SOP

### 4.1 新图进入系统

每批新图先进入待处理目录，例如：

- `incoming/images`

### 4.2 模型 A 自动出点

用缝纫点检测模型生成点预测。

当前标准要求：

- 点预测结果应直接保存为 **CAB-F 母格式**
- 不再把 LabelMe 点标注作为长期中间真值格式
- 如果只是为了兼容旧工具链，才临时导出 LabelMe 点标注

推荐命令：

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point.tools.batch_infer `
  --input_dir <images_dir> `
  --output_dir <point_prediction_dir> `
  --model <sew_point_onnx> `
  --threshold 0.3 `
  --output_format master
```

说明：

- `--output_format master` 为当前推荐默认模式
- 输出 JSON 结构应为：
  - `points`
  - `edges`（此阶段为空）
  - `metadata`
- `metadata.source` 推荐保留模型来源，例如 `sew_point_batch_infer`

### 4.3 人工修点

在 `img_tools` 中：

- 打开 CAB-F 点编辑器
- 修正漏点、误点、偏移点

保存后得到母格式的点版标注：

- `points` 已确认
- `edges` 先为空

### 4.4 模型 B 自动出边

基于已确认的点，运行模型 B 预测边。

当前标准要求：

- 输入应为母格式 JSON，且 `points` 已存在
- 输出仍应保存为母格式 JSON
- 模型预测得到的边只作为“待审核结果”，不能直接当最终真值

推荐命令：

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point_conntect.batch_predict `
  --image_dir <images_dir> `
  --annotation_dir <point_annotation_dir> `
  --model_path <sew_point_conntect_pth> `
  --output_annotation_dir <edge_prediction_dir>
```

说明：

- `annotation_dir` 推荐传入“母格式点标注目录”
- 预测结果会带上 `edges`
- `metadata.source` 应标记为类似 `sew_point_conntect_batch_predict`

### 4.5 人工修边

在 `img_tools` 中：

- 打开 CAB-F 连边标注器
- 删除错边
- 补上漏边
- 修正复杂区域

保存后得到最终母标注真值。

### 4.6 执行校验

每次入库后，必须运行 CAB-F 校验工具。

检查重点：

- 图片与 JSON 是否一一对应
- 点 ID 是否唯一
- 边是否引用了存在的点
- 是否出现自环
- 是否出现重复边
- 一个点的度数是否超过 2
- 是否存在 0 点 / 1 点 / 0 边样本

### 4.7 导出训练集

校验通过后，分别导出：

- 模型 A 训练集
- 模型 B 训练集

---

## 5. 模型辅助标注策略

### 点标注

推荐：

- 模型预标
- 人工快速修正

点标注可以高度依赖模型。

### 边标注

推荐：

- 模型预标
- 人工审核修正

边标注不能直接把模型输出当真值，尤其是多条线路混在一起时。

---

## 6. 真值与预测的边界

下面几类数据必须区分：

### 真值母标注

允许进入训练的数据，只能来自：

- 人工确认后的 `master_annotations`

### 模型预测结果

只能作为中间产物：

- 点预测
- 边预测
- 批量预测结果

它们不能直接覆盖母标注。

补充约定：

- 点预测中间结果推荐存放在 `predictions/points`
- 边预测中间结果推荐存放在 `predictions/edges`
- 真值母标注始终只维护在 `master_annotations/annotations`

---

## 7. 每次新增数据的固定动作

以后每一批新数据都按下面顺序执行：

1. 模型 A 出点
2. 人工修点
3. 模型 B 出边
4. 人工修边
5. 保存母标注
6. 运行校验
7. 导出模型 A / 模型 B 训练集
8. 再进入训练

当前推荐的“正式闭环”是：

`256x256 图片 -> 模型 A 出母格式点 -> 人工修点 -> 模型 B 出母格式边 -> 人工修边 -> 校验 -> 导出 -> 训练`

---

## 8. 推荐工具入口

### GUI

`img_tools` 菜单：

- `工具 -> CAB-F 连边标注器`
- `工具 -> CAB-F 缝纫点数据筛选`
- `工具 -> CAB-F 数据集校验与导出`

### CLI

```powershell
cd D:\project\tianwei\img_tools

python cli.py cabf validate --image-dir <images> --annotation-dir <annotations>
python cli.py cabf export-model-a --image-dir <images> --annotation-dir <annotations> --output-dir <model_a_export_root>
python cli.py cabf export-model-b --image-dir <images> --annotation-dir <annotations> --output-dir <model_b_export_root>
```

导出后目录结构固定为：

```text
<model_a_export_root>/
  images/
  annotations/
  error/

<model_b_export_root>/
  images/
  annotations/
  error/
```

训练命令示例：

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point.train `
  --img_dir <model_a_export_root>\images `
  --ann_dir <model_a_export_root>\annotations `
  --save_dir <sew_point_train_out>

python -m sew_point_conntect.train `
  --image_dir <model_b_export_root>\images `
  --annotation_dir <model_b_export_root>\annotations `
  --save_dir <sew_point_conntect_train_out>
```

---

## 9. 最终要求

所有训练数据必须满足：

- 有统一的母格式
- 能通过校验
- 能稳定导出
- 能追溯来源

如果一份数据不能明确回答：

- 点从哪里来
- 边从哪里来
- 是否经过人工确认

那么它不应直接进入正式训练集。

补充：

- 当前正式流程以“母格式贯穿”作为默认方案
- `sew_point` 的批量 `256x256` 小图推理已经可以直接输出母格式
- 后续维护时，应优先保证所有新样本都先进入母格式，再进入人工审核、导出和训练
