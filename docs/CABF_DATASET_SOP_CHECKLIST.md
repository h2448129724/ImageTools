# CAB-F 数据集现场执行版清单

适用范围：

- `256x256` 小图
- `sew_point` 点检测
- `sew_point_conntect` 连边预测
- `img_tools` CAB-F 校验、人工修订、导出

目标：

- 全流程只维护一套 **母格式真值**
- 模型预测结果只作为中间产物
- 最终训练集统一从母格式导出

---

## 1. 目录准备

建议固定使用：

```text
CAB-F/
  master_annotations/
    images/
    annotations/
  predictions/
    points/
    edges/
  model_a_export/
  model_b_export/
```

说明：

- `master_annotations/images`：正式图片源
- `master_annotations/annotations`：正式母标注真值源
- `predictions/points`：点模型中间结果
- `predictions/edges`：边模型中间结果
- `model_a_export`：导出给 `sew_point` 训练
- `model_b_export`：导出给 `sew_point_conntect` 训练

---

## 2. 点模型出点

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point.tools.batch_infer `
  --input_dir <CAB-F>\master_annotations\images `
  --output_dir <CAB-F>\predictions\points `
  --model <sew_point_onnx> `
  --threshold 0.3 `
  --output_format master
```

检查点：

- 输出 JSON 是母格式
- `points` 有值
- `edges` 为空
- `metadata.source = sew_point_batch_infer`

---

## 3. 人工修点

在 `img_tools` 中打开 CAB-F 点编辑相关工具，处理：

- 漏点
- 误点
- 偏移点

处理完成后：

- 保存为母格式
- 结果进入 `master_annotations/annotations`

---

## 4. 边模型出边

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point_conntect.batch_predict `
  --image_dir <CAB-F>\master_annotations\images `
  --annotation_dir <CAB-F>\master_annotations\annotations `
  --model_path <sew_point_conntect_pth> `
  --output_annotation_dir <CAB-F>\predictions\edges
```

检查点：

- 输入是母格式点标注
- 输出仍是母格式
- `edges` 已生成
- `metadata.source = sew_point_conntect_batch_predict`

---

## 5. 人工修边

在 `img_tools` 中打开：

- `工具 -> CAB-F 连边标注器`

处理：

- 删除错边
- 补漏边
- 修复杂区域

处理完成后：

- 保存为最终母标注
- 结果回写到 `master_annotations/annotations`

---

## 6. 数据校验

```powershell
cd D:\project\tianwei\img_tools

python cli.py cabf validate `
  --image-dir <CAB-F>\master_annotations\images `
  --annotation-dir <CAB-F>\master_annotations\annotations
```

必须确认：

- 图片和 JSON 一一对应
- `points[].id` 唯一
- `edges[].src/dst` 引用存在的点
- 无自环
- 无重复边
- 无明显异常样本

说明：

- 有问题的样本先修，再继续下一步

---

## 7. 导出训练集

导出模型 A：

```powershell
python cli.py cabf export-model-a `
  --image-dir <CAB-F>\master_annotations\images `
  --annotation-dir <CAB-F>\master_annotations\annotations `
  --output-dir <CAB-F>\model_a_export
```

导出模型 B：

```powershell
python cli.py cabf export-model-b `
  --image-dir <CAB-F>\master_annotations\images `
  --annotation-dir <CAB-F>\master_annotations\annotations `
  --output-dir <CAB-F>\model_b_export
```

导出后检查：

```text
model_a_export/
  images/
  annotations/
  error/

model_b_export/
  images/
  annotations/
  error/
```

说明：

- 有问题样本会进入 `error/`
- 正式训练只使用 `images/` 和 `annotations/`

---

## 8. 训练新权重

训练模型 A：

```powershell
cd D:\project\tianwei\train_model\modules

python -m sew_point.train `
  --img_dir <CAB-F>\model_a_export\images `
  --ann_dir <CAB-F>\model_a_export\annotations `
  --save_dir <sew_point_train_out>
```

训练模型 B：

```powershell
python -m sew_point_conntect.train `
  --image_dir <CAB-F>\model_b_export\images `
  --annotation_dir <CAB-F>\model_b_export\annotations `
  --save_dir <sew_point_conntect_train_out>
```

---

## 9. 现场判断原则

只要不满足下面任一条件，就不要进正式训练：

- 点来源清楚
- 边来源清楚
- 已人工确认
- 已通过校验
- 已从母格式正式导出

---

## 10. 最短闭环

正式执行时，按这个顺序走：

`图片 -> 点预测(母格式) -> 人工修点 -> 边预测(母格式) -> 人工修边 -> 校验 -> 导出 -> 训练`
