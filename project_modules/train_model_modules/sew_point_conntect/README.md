# sew_point_conntect

针点连边分类模块。

当前已接入训练工作台，子功能包括：

- 训练
- 单样本推理
- 批量预测
- 可视化评估

## 数据要求

训练输入显式分为：

- `--image_dir`：图片目录
- `--annotation_dir`：边标注 `json` 目录

不再推荐通过单一 `data_dir` 自动推断内部结构。

## 训练

```bash
cd modules/sew_point_conntect
python train.py \
  --image_dir /data/sew_point_connect/images \
  --annotation_dir /data/sew_point_connect/annotations \
  --save_dir ../../artifacts/checkpoints/sew_point_conntect_train \
  --epochs 300 \
  --batch_size 16 \
  --hidden_dim 128 \
  --num_layers 3
```

常用参数：

- `--lr`
- `--weight_decay`
- `--warmup_epochs`
- `--val_ratio`
- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--k_neighbors`
- `--patch_width`
- `--patch_height`

## 单样本推理

```bash
cd modules/sew_point_conntect
python infer.py \
  --json_path /data/sew_point_connect/sample.json \
  --image_path /data/sew_point_connect/sample.png \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth
```

## 批量预测

```bash
cd modules/sew_point_conntect
python batch_predict.py \
  --image_dir /data/sew_point_connect/images \
  --annotation_dir /data/sew_point_connect/annotations \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth \
  --output_annotation_dir /data/sew_point_connect_pred/annotations \
  --vis_dir /data/sew_point_connect_pred/vis
```

## 可视化评估

```bash
cd modules/sew_point_conntect
python visualize_eval.py \
  --json_path /data/sew_point_connect/sample.json \
  --image_path /data/sew_point_connect/sample.png \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth \
  --out_dir /data/sew_point_connect_eval
```

## 输出说明

- 训练权重可保存到 `--save_dir`
- GUI 运行历史统一写入 `artifacts/runs/`
- 如果在 GUI 中右键“删除”历史，只会移动到项目 `.trash/`
