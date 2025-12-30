# 机器学习课程作业说明文档

## 目录

- [机器学习课程作业说明文档](#机器学习课程作业说明文档)
  - [目录](#目录)
  - [一、项目说明](#一项目说明)
  - [二、目录结构说明](#二目录结构说明)
  - [三、我们新增与修改的内容](#三我们新增与修改的内容)
    - [1. 新增 / 修改的模型](#1-新增--修改的模型)
    - [2. 新增 Runner](#2-新增-runner)
    - [3. 新增实验运行脚本（Bash）](#3-新增实验运行脚本bash)
  - [四、实验运行说明](#四实验运行说明)
    - [1. 环境配置](#1-环境配置)
    - [2. Baseline 实验](#2-baseline-实验)
    - [3. StagewiseGCN 实验](#3-stagewisegcn-实验)
    - [4. 改进模型（TargetGCN）](#4-改进模型targetgcn)
    - [5. 消融实验](#5-消融实验)
  - [五、实验结果说明](#五实验结果说明)
  - [六、版权与致谢说明](#六版权与致谢说明)
  - [七、说明](#七说明)

## 一、项目说明

本项目基于课程提供的原始代码仓库进行实验与扩展，用于完成机器学习课程作业中的推荐系统相关实验。

- **`README_origin.md`**

    原始仓库的 README 文件，**完整保留**，重命名为 `README_origin.md`。

- **`README.md`（本文档）**

  内容包括：
  
  - 我们新增与修改的内容
  - 实验如何运行
  - 实验结果如何复现

> 本仓库**未上传训练得到的模型文件**（体积较大），但已完整保留实验日志，可用于结果复现与核查。

## 二、目录结构说明

```text
.
├── data/       # 数据集目录（我们已经准备了我们的作业中用到的数据集）
├── docs/
├── log/        # 实验日志记录
│   ├── AblationStagewiseGCN/   # 消融实验日志
│   ├── BPRMF/                  # 基线模型日志
│   ├── LightGCN/               # 基线模型日志
│   ├── NeuMF/                  # 基线模型日志
│   ├── StagewiseGCN/           # 复现模型日志
│   └── TargetGCN/              # 改进模型日志
├── src/
│   ├── models/
│   │   └── general/
│   │       ├── StagewiseGCN.py             # 我们所复现的模型
│   │       ├── TargetGCN.py                # 我们所改进的模型
│   │       └── AblationStagewiseGCN.py     # StagewiseGCN 的消融版本
│   └── helpers/
│       ├── StagewiseRunner.py              # 对应 StagewiseGCN 的训练 Runner
│       └── AblationStagewiseRunner.py      # 对应消融实验的 Runner
├── run_baseline.sh             # Baseline 模型运行脚本
├── run_hparam_1.sh             # 超参数实验脚本（组 1，针对 Grocery and Gourmet Food 数据集）
├── run_hparam_2.sh             # 超参数实验脚本（组 2，针对 ML_1MTOPK 数据集）
├── run_ablation.sh             # 消融实验脚本
├── run_improved_model.sh       # 改进模型（TargetGCN）运行脚本
├── requirements.txt            # 改进后的 requirements.txt
├── LICENSE                     # 保留原仓库的许可证
├── README_origin.md            # 保留原仓库的 README.md
└── README.md                   # 本文件
```

## 三、我们新增与修改的内容

### 1. 新增 / 修改的模型

- **StagewiseGCN**

  - 文件：`src/models/general/StagewiseGCN.py`
  - 复现论文：*Joint Multi-Grained Popularity-Aware Graph
Convolution Collaborative Filtering for
Recommendation*，[访问链接](https://dl.acm.org/doi/10.1145/3397271.3401163)

- **AblationStagewiseGCN**

  - 文件：`src/models/general/AblationStagewiseGCN.py`
  - 用于消融实验，仅保留某一个阶段（Stage 1 / 2 / 3）

- **TargetGCN**

  - 文件：`src/models/general/TargetGCN.py`
  - 我们改进的模型，在奇数层与偶数层使用不同的编码器

### 2. 新增 Runner

- `StagewiseRunner.py`
- `AblationStagewiseRunner.py`

用于适配阶段式训练流程与消融实验逻辑。

### 3. 新增实验运行脚本（Bash）

所有 `.sh` 文件均为**我们自行编写**，用于一键复现实验：

| 脚本名                     | 作用                                       |
| ----------------------- | ---------------------------------------- |
| `run_baseline.sh`       | 运行 Baseline 模型（BPRMF / LightGCN / NeuMF） |
| `run_hparam_1.sh`       | 第一组超参数实验                                 |
| `run_hparam_2.sh`       | 第二组超参数实验                                 |
| `run_ablation.sh`       | StagewiseGCN 消融实验                        |
| `run_improved_model.sh` | 改进模型 TargetGCN 实验                        |

---

## 四、实验运行说明

### 1. 环境配置

建议使用 Python 虚拟环境、并使用 cuda 运行（因为原框架含有必须使用 cuda 的语句）：

```bash
pip install -r requirements.txt
```

已经验证过可以运行的环境（理论上只要不是太低的 torch 和 python 版本都可以运行）：

```txt
Python == 3.11.9
torch==2.9.1
tqdm==4.66.1
pandas==2.3.3
numpy==2.4.0
scikit-learn==1.8.0
scipy==1.16.3
PyYAML==6.0.3
```

### 2. Baseline 实验

```bash
bash run_baseline.sh
```

运行内容包括：

* BPRMF
* LightGCN
* NeuMF

### 3. StagewiseGCN 实验

```bash
bash run_hparam_1.sh
bash run_hparam_2.sh
```

用于不同超参数组合的实验。

### 4. 改进模型（TargetGCN）

```bash
bash run_improved_model.sh
```

### 5. 消融实验

```bash
bash run_ablation.sh
```

- 对 StagewiseGCN 的不同阶段进行消融
- 分别仅保留 Stage 1 / Stage 2 / Stage 3
- 用于验证多阶段结构的有效性

## 五、实验结果说明

- 所有实验的**完整训练日志**已保存在 `log/` 目录下，例如：

```text
log/
├── StagewiseGCN/
├── TargetGCN/
├── AblationStagewiseGCN/
└── ...
```

- 日志中包含：

  - 每个 Stage、Epoch 的 loss
  - 验证集 HR / NDCG
  - 最优模型标记（`*`）

> **由于模型文件体积过大，未上传 `.pt` / `.pth` 权重文件**，但所有实验结果均可通过日志复现与核查。

## 六、版权与致谢说明

- 本项目基于原始仓库进行课程作业实验
- 原始 README 已完整保存在 `README_origin.md`
- 所有新增模型、Runner 与实验脚本均为我们自行完成
- 本仓库仅用于课程学习与作业提交，不作商业用途

## 七、说明

> 若只需验证实验结果，请直接查看 `log/` 目录；
> 若需复现实验，请按第四部分运行对应 `.sh` 脚本。