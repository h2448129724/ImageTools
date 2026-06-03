# CAB-F 一体化仓库迁移方案

适用场景：

- 当前 `img_tools` 与 `train_model` 主要由同一人维护和使用
- 目标是让 CAB-F 的“数据生产 -> 校验 -> 导出 -> 训练”全流程更顺手
- 优先追求实用、稳定、少切换上下文，而不是多人协作隔离

---

## 1. 结论

推荐把当前两个 Git 仓库收敛为：

- **一个总仓库**
- **仓库内部保留两个功能模块**
- **新增一层共享公共层**

不建议把所有代码直接平铺混在一起。

推荐形态是：

```text
cabf_pipeline/
  apps/
    img_tools/
    train_model/
  shared/
    cabf/
  docs/
  scripts/
  examples/
```

这样做的目标是：

- 继续保留“图像工具”和“训练模块”的边界
- 把母格式、SOP、路径约定、公共 JSON 处理统一到一个地方
- 让整个项目以“同一版本的完整流程”方式演进

---

## 2. 为什么现在适合合库

你当前的实际使用方式已经说明两件事：

1. `img_tools` 和 `train_model` 不是独立使用的
2. 它们通过 CAB-F 母格式和 SOP 形成了固定闭环

已经存在的强耦合点包括：

- CAB-F 母格式 `points + edges + metadata`
- `sew_point` 输出母格式点标注
- `sew_point_conntect` 读取母格式点标注并输出母格式边标注
- `img_tools` 承担校验、人工修订、导出
- 导出结果直接喂给训练脚本

也就是说，你现在维护的不是两个独立产品，而是一条完整生产线。

---

## 3. 推荐目录结构

推荐最终目录：

```text
cabf_pipeline/
  .gitignore
  README.md

  apps/
    img_tools/
      main.py
      cli.py
      pyproject.toml
      core/
      gui/
      tests/

    train_model/
      trainer_gui/
      modules/
        sew_point/
        sew_point_conntect/
        segmentation/
        yolo/
      artifacts/
      docs/

  shared/
    cabf/
      __init__.py
      schema.py
      io.py
      validation.py
      export.py
      paths.py

  docs/
    CABF_MASTER_SCHEMA.md
    CABF_DATASET_SOP.md
    CABF_DATASET_SOP_CHECKLIST.md
    CABF_MONOREPO_MIGRATION_PLAN.md

  scripts/
    cabf_predict_points.ps1
    cabf_predict_edges.ps1
    cabf_validate.ps1
    cabf_export.ps1
    cabf_train.ps1

  examples/
    cabf_folder_layout.md
```

---

## 4. 各层职责

### 4.1 `apps/img_tools`

保留：

- GUI
- 标注器
- 点边一体编辑器
- 数据集校验与导出
- CLI 工具入口

不建议继续把“母格式核心定义”只放在这里独占。

### 4.2 `apps/train_model`

保留：

- `sew_point`
- `sew_point_conntect`
- `segmentation`
- `yolo`
- 训练 GUI
- 推理与训练脚本

不建议继续在这里维护另一套与母格式平行的协议逻辑。

### 4.3 `shared/cabf`

这是合库后最关键的一层，建议集中放：

- 母格式版本号
- JSON 标准字段
- LabelMe -> 母格式转换
- 母格式归一化
- 校验逻辑
- 导出逻辑
- 路径约定与命名规则

以后只要改母格式，优先改这一层。

### 4.4 `docs`

建议把当前散落文档统一收口到仓库根下 `docs/`：

- 母格式说明
- 正式 SOP
- 现场执行版清单
- 迁移说明

这样你以后不会再出现“文档在这个库，代码在那个库”的来回跳转。

### 4.5 `scripts`

建议补一批流程入口脚本：

- 点预测
- 边预测
- 校验
- 导出
- 训练

作用不是替代 Python 模块，而是减少你现场输入命令和记路径的负担。

---

## 5. 建议优先抽离的共享代码

第一批最值得抽到 `shared/cabf` 的内容：

1. `img_tools/core/cabf_dataset.py`
2. 母格式常量，如 `MASTER_SCHEMA_VERSION`
3. LabelMe 点标注转换逻辑
4. `points` / `edges` 归一化逻辑
5. 导出模型 A / 模型 B 的逻辑
6. 样本名、图片名、JSON 文件名的匹配规则

第二批再考虑抽：

1. 点边编辑器使用的空标注模板
2. 预测结果 metadata 约定
3. `predictions/points`、`predictions/edges` 命名约定

---

## 6. 推荐迁移顺序

### 阶段 1：逻辑先统一，不急着搬代码

目标：

- 保持两个仓库先能继续工作
- 先把“共享定义”从概念上固定下来

动作：

1. 固定母格式说明文档
2. 固定现场 SOP
3. 固定文件命名约定
4. 固定 `metadata` 约定

### 阶段 2：建立总仓库骨架

目标：

- 新建一体化仓库目录
- 先让结构成立，不急着把所有 import 一次改完

动作：

1. 创建 `cabf_pipeline/`
2. 建立 `apps/`、`shared/`、`docs/`、`scripts/`
3. 把现有两个仓库以子目录形式放进去

### 阶段 3：抽共享层

目标：

- 把最核心的 CAB-F 公共逻辑搬进 `shared/cabf`

动作：

1. 先抽 schema / io / validation / export
2. `img_tools` 改成优先调用共享层
3. `train_model` 中涉及母格式的脚本改成调用共享层

### 阶段 4：补流程脚本

目标：

- 把常用链路变成一键可执行入口

建议至少补这几个：

- `scripts/cabf_predict_points.ps1`
- `scripts/cabf_predict_edges.ps1`
- `scripts/cabf_validate.ps1`
- `scripts/cabf_export.ps1`
- `scripts/cabf_train.ps1`

### 阶段 5：最后再决定是否彻底废弃旧仓库

目标：

- 等新仓库跑顺后，再决定是否长期停用旧入口

注意：

- 在未稳定前，不建议立刻删除任何旧目录
- 可以保留旧仓库作为过渡期参考

---

## 7. 现场最实用的收益

如果按这个方案收口，你会直接获得这些好处：

- SOP、代码、schema 在一个版本里演进
- 不用在两个 Git 仓库之间来回切换
- 推理、标注、校验、导出、训练路径统一
- 改命名规则时不容易漏改另一边
- 后续补一键脚本会非常顺手

---

## 8. 不建议的做法

下面几种方式不推荐：

### 8.1 所有代码直接混成一个目录

问题：

- 后期更乱
- 工具代码和训练代码边界消失

### 8.2 继续双仓，但长期手动同步 schema

问题：

- 很容易漂移
- 你会不断遇到“这个库改了，另一个没跟上”

### 8.3 一步到位大重构

问题：

- 风险高
- 现场流程容易断

更稳妥的是：

- 先确定结构
- 再一点点抽共享层
- 最后再统一入口

---

## 9. 推荐最终目标

你这个项目最合适的终局不是“两个独立工具”，而是：

- 一个 CAB-F 一体化流程仓库
- 两个主要应用模块
- 一层共享 CAB-F 核心层
- 一套正式 SOP 和现场清单
- 一组一键执行脚本

简化表达就是：

`一个仓库 + 两个应用 + 一个共享层`

---

## 10. 下一步建议

如果继续往下做，推荐顺序是：

1. 先确定一体化仓库根目录名字
2. 设计 `shared/cabf` 的首批文件
3. 先把 `img_tools/core/cabf_dataset.py` 拆成共享层
4. 再让 `sew_point` / `sew_point_conntect` 改用共享层
5. 最后补一键流程脚本

如果要尽量少折腾现有流程，那么最好的第一步不是“直接搬家”，而是：

- **先做共享层设计**
- **再做仓库结构迁移**
