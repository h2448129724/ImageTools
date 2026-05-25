# CAB-F 母标注格式与版本说明

当前生效版本：`1.2`

对应代码常量：

- [D:\project\tianwei\img_tools\core\cabf_dataset.py](</d:/project/tianwei/img_tools/core/cabf_dataset.py>) 中的 `MASTER_SCHEMA_VERSION = "1.2"`

---

## 1. 目标

CAB-F 母标注用于统一承载：

- 缝纫点
- 连边关系
- 来源与追踪信息

模型 A 与模型 B 都不直接维护独立真值源，而是从这套母标注导出。

---

## 2. 当前标准字段

```json
{
  "schema_version": "1.2",
  "sample_id": "sample_xxx",
  "image_path": "sample_xxx.png",
  "image_size": {
    "width": 256,
    "height": 256
  },
  "roi": null,
  "spacing_hint": null,
  "points": [],
  "edges": [],
  "segments": [],
  "metadata": {}
}
```

---

## 3. 字段解释

### 3.1 顶层字段

#### `schema_version`

- 类型：`string`
- 当前值：`"1.2"`
- 作用：标识母格式版本

#### `sample_id`

- 类型：`string`
- 作用：样本唯一标识
- 建议：与图片文件名去扩展名一致

#### `image_path`

- 类型：`string`
- 作用：图片路径或图片文件名
- 建议：长期优先保存为图片文件名或相对路径

#### `image_size`

- 类型：`object`
- 子字段：
  - `width: int`
  - `height: int`

#### `roi`

- 类型：`object | null`
- 作用：可选，保留 ROI 信息

#### `spacing_hint`

- 类型：`number | null`
- 作用：可选，记录针距先验或人工估计

#### `points`

- 类型：`array`
- 作用：保存缝纫点

#### `edges`

- 类型：`array`
- 作用：保存点与点之间的连接关系

#### `segments`

- 类型：`array`
- 作用：预留给后续路径段或顺序标注
- 当前可以为空

#### `metadata`

- 类型：`object`
- 作用：保存来源、模型、人工校验等附加信息

---

## 4. `points` 结构

每个点的标准格式：

```json
{
  "id": 0,
  "x": 14.0,
  "y": 38.0,
  "score": 0.78,
  "source": "manual"
}
```

字段说明：

- `id`
  - `int`
  - 在当前样本内唯一

- `x`, `y`
  - `float`
  - 图像坐标

- `score`
  - `float`
  - 可选置信度，人工点可设为 `1.0`

- `source`
  - `string`
  - 推荐值：
    - `manual`
    - `model`
    - `labelme_point`
    - `point_editor`

---

## 5. `edges` 结构

每条边的标准格式：

```json
{
  "edge_id": "edge_0001",
  "src": 0,
  "dst": 1,
  "label": 1,
  "source": "manual"
}
```

可选预测边示例：

```json
{
  "edge_id": "pred_edge_0001",
  "src": 0,
  "dst": 1,
  "score": 0.99,
  "label": 1,
  "source": "gnn_predict"
}
```

字段说明：

- `edge_id`
  - `string`
  - 当前样本内唯一

- `src`, `dst`
  - `int`
  - 引用 `points[].id`

- `label`
  - `int`
  - 当前默认只用 `1`

- `score`
  - `float`
  - 预测边可带
  - 人工真值边可不带

- `source`
  - `string`
  - 推荐值：
    - `manual`
    - `gnn_predict`
    - `imported`

---

## 6. `metadata` 约定

`metadata` 没有强制固定死，但推荐包含：

```json
{
  "source": "img_tools_edge_editor",
  "origin_json": "xxx.json",
  "point_count": 9,
  "edge_count": 8
}
```

对不同场景，推荐字段如下。

### 人工点编辑后

```json
{
  "source": "point_editor"
}
```

### 人工连边后

```json
{
  "source": "img_tools_edge_editor",
  "point_count": 9,
  "edge_count": 8
}
```

### 模型预测后

```json
{
  "source": "sew_point_conntect_batch_predict",
  "origin_json": "xxx.json",
  "predicted_edge_count": 8,
  "model_path": "best.pth"
}
```

---

## 7. 校验规则

当前母格式至少应满足：

1. `sample_id` 存在
2. `image_path` 存在
3. `image_size.width/height` 存在
4. `points[].id` 唯一
5. `edges[].src/dst` 引用存在的点
6. 不允许自环
7. 不允许重复边
8. 一个点的度数原则上不应超过 `2`

说明：

- “度数超过 2” 当前在校验器中是 **warning**，不是硬错误
- 这样方便先保存模型预测结果，再人工修正

---

## 8. 与旧格式的关系

### 8.1 LabelMe 点标注

旧格式：

- `version`
- `shapes`
- `shape_type = point`

可以自动转换为母格式：

- `points` 有值
- `edges` 为空
- `metadata.source = labelme_point`

### 8.2 旧版 `1.0 / 1.1`

如果已有：

- `points`
- `edges`
- `metadata`

通常只需要规范化后升级成 `1.2`。

---

## 9. 版本演进规则

### 当前正式版本

- `1.2`

### 升级原则

只有在下面几种情况才升级版本：

1. 顶层字段结构变化
2. `points` / `edges` 语义变化
3. 导出或训练依赖的关键字段变化

### 升级后必须做的事

1. 更新 `core/cabf_dataset.py` 里的 `MASTER_SCHEMA_VERSION`
2. 更新本文件
3. 更新 SOP
4. 更新 GUI / CLI 的导出与校验逻辑

---

## 10. 推荐结论

以后所有 CAB-F 真值数据统一按 `1.2` 存。

如果是：

- 模型预测结果
- 中间转换结果
- 旧点标注迁移结果

也建议尽快规范到这套结构，再进入人工复核和训练流程。
