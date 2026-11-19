# UCI HAR 小型 1D-CNN 实验

本目录实现基于 **9 路惯性时间序列** （9通道、每通道128个时间点的惯性信号）的六类人体活动识别。模型输入是传感器窗口，不是项目中用于 GRRF/SVM 的 561 维工程特征。实验先复用项目已有的按参与者划分的五折文件完成交叉验证，再按事先确定的配置用全部官方训练集训练最终模型，并在官方测试集评估一次。

## 1. 文件与目录

路径均相对于项目根目录。

| 路径 | 作用 |
| --- | --- |
| `src/cnn1d/data.py` | 读取 9 路信号、活动标签和参与者编号；可单独运行以检查形状。 |
| `src/cnn1d/train.py` | 定义网络、验证现有五折文件、训练、早停、评估和保存结果。 |
| `src/cnn1d/train_final_evaluate.py` | 按已选配置训练最终模型，随后评估官方测试集。 |
| `src/cnn1d/README.md` | 本实验的实现和结果说明。 |
| `data/raw/UCI HAR Dataset/` | UCI HAR 原始发布数据。 |
| `results/grrf_cv5_splits.csv` | 已有的训练集五折分配；CNN 直接读取，不重新生成。 |
| `results/cnn1d/` | 本实验的输出目录。 |

## 2. 数据读取

`data.py` 分别读取官方 `train/` 和 `test/` 目录中的文件。9 个通道按以下固定顺序堆叠：

1. `body_acc_x`、`body_acc_y`、`body_acc_z`
2. `body_gyro_x`、`body_gyro_y`、`body_gyro_z`
3. `total_acc_x`、`total_acc_y`、`total_acc_z`

每个通道包含 128 个时间点。`load_split(split)` 返回 `signals, labels, subjects`：

| 数据 | 信号形状 | 标签形状 | 参与者编号形状 |
| --- | --- | --- | --- |
| 官方训练集 | `(7352, 9, 128)` | `(7352,)` | `(7352,)` |
| 官方测试集 | `(2947, 9, 128)` | `(2947,)` | `(2947,)` |

信号采用 `float32`；标签和参与者编号采用 `int64`。原始标签 `1–6` 在读取时减 1，转换为交叉熵损失需要的 `0–5`；参与者编号保持原值。类别顺序如下：

| 模型标签 | 原始标签 | 活动 |
| ---: | ---: | --- |
| 0 | 1 | WALKING（步行） |
| 1 | 2 | WALKING_UPSTAIRS（上楼） |
| 2 | 3 | WALKING_DOWNSTAIRS（下楼） |
| 3 | 4 | SITTING（坐） |
| 4 | 5 | STANDING（站立） |
| 5 | 6 | LAYING（躺） |

读取程序检查信号是否为 `(N, 9, 128)`、三组数据的样本数是否相同，以及标签是否处于合法范围。当前没有额外进行标准化或数据增强。

## 3. 网络结构

PyTorch 的 `Conv1d` 输入格式为 `(batch, channels, time)`，因此单条数据为 `(9, 128)`。当前模型是**三层一维卷积**，并非两层卷积版本。

| 顺序 | 层 | 单条数据的输出形状 |
| ---: | --- | --- |
| 1 | 输入 | `(9, 128)` |
| 2 | `Conv1d(9, 32, kernel_size=7, padding=3)` → BatchNorm → ReLU | `(32, 128)` |
| 3 | `MaxPool1d(2)` | `(32, 64)` |
| 4 | `Conv1d(32, 64, kernel_size=5, padding=2)` → BatchNorm → ReLU | `(64, 64)` |
| 5 | `MaxPool1d(2)` | `(64, 32)` |
| 6 | `Conv1d(64, 64, kernel_size=3, padding=1)` → BatchNorm → ReLU | `(64, 32)` |
| 7 | `AdaptiveAvgPool1d(1)` → Flatten | `(64,)` |
| 8 | `Dropout(0.3)` → `Linear(64, 6)` | 6 个类别 logit |

全局平均池化将每个通道沿时间维汇总。最终输出是六个未经 softmax 的 logit，直接交给 `CrossEntropyLoss`；预测类别取最大 logit 的索引。

## 4. 五折训练与复现

`train.py` 只加载**官方训练集**进行五折实验。它读取 `results/grrf_cv5_splits.csv` 的 `row_index_python`、`subject`、`validation_fold` 三列，并检查行号、参与者编号、折号及参与者归属。对于第 `k` 折，`validation_fold == k` 的样本作验证，其余样本作训练；同一参与者不会同时出现在两侧。脚本不会重新生成五折划分，也不会在此阶段读取官方测试集。

每折从头初始化一个独立模型。默认设置：

| 参数 | 默认值 / 规则 |
| --- | --- |
| 优化器 | Adam，学习率 `0.001` |
| 损失 | 六类交叉熵 |
| Batch size | `128` |
| 最多训练轮数 | `40` |
| 早停 | 验证集 Macro-F1 连续 `7` 轮未提高时停止 |
| 模型选择 | 保存验证集 Macro-F1 最高轮次的权重 |
| 基础随机种子 | `42`；第 `k` 折使用 `42 + k` |
| 设备 | 有 CUDA 时使用 CUDA，否则使用 CPU |

固定 Python、NumPy、PyTorch 和 DataLoader 的随机种子；DataLoader 使用独立生成器且 `num_workers=0`。同时启用 PyTorch 确定性算法，关闭 cuDNN benchmark，并设置确定性相关环境变量。配置记录在各折 `run_config.json` 中，包含五折文件的 SHA-256 校验值。在当前 CPU 环境下，第 1、2 折曾重复运行，模型权重与逐轮训练记录的哈希值均一致；跨设备或不同软件版本不保证逐位相同。

在**项目根目录**运行：

```powershell
python src/cnn1d/data.py
python src/cnn1d/train.py --fold 1
python src/cnn1d/train.py --fold 2
python src/cnn1d/train.py --fold 3
python src/cnn1d/train.py --fold 4
python src/cnn1d/train.py --fold 5
```

省略 `--fold` 可顺序重跑全部五折：`python src/cnn1d/train.py`。可用 `--epochs`、`--batch-size`、`--patience`、`--seed` 覆盖默认值。所需 Python 包见项目根目录 `requirements.txt`，其中包含 PyTorch。

## 5. 输出文件与指标

每折写入 `results/cnn1d/fold_k/`：

| 文件 | 内容 |
| --- | --- |
| `model.pt` | 该折验证 Macro-F1 最佳轮次的模型权重。 |
| `history.csv` | 每轮 `train_loss`、`val_loss`、`val_accuracy`、`val_macro_f1`。 |
| `training_curves.png` | 训练/验证交叉熵损失、验证 Accuracy/Macro-F1；竖虚线标记最佳轮次。 |
| `metrics.json` | 最佳权重对应的验证 Accuracy、Macro-F1、六类召回率、轮次、设备和样本数。**不包含训练耗时**。 |
| `confusion_matrix.csv` | 6×6 混淆矩阵：行是真实类别，列是预测类别，顺序为上表模型标签 `0–5`。 |
| `run_config.json` | 随机种子、确定性选项、超参数、软件版本和五折文件校验值。 |

五折文件齐全时，脚本还会更新 `results/cnn1d/fold_metrics.csv`（逐折指标）和 `results/cnn1d/cv_summary.csv`（五折算术平均与样本标准差）。训练耗时仅在终端打印，不写入 `metrics.json`。

### 五折汇总文件

两个文件均位于 `results/cnn1d/`，指标数值以 `0–1` 的比例保存；下文结果表将其换算为百分比。

| 文件 | 字段与含义 |
| --- | --- |
| `fold_metrics.csv` | 每折一行。`fold` 是折号；`accuracy`、`macro_f1` 是该折**最佳模型**在对应验证集上的指标；`best_epoch` 是验证 Macro-F1 最高的轮次；`epochs_trained` 是包含早停等待在内实际运行的轮数；`train_samples`、`n_val_samples` 分别是训练和验证样本数。 |
| `cv_summary.csv` | 对 `fold_metrics.csv` 中五折的 `accuracy` 和 `macro_f1` 分别计算算术平均（`mean` 行）与样本标准差（`std` 行）。每折权重相同，不按验证样本数加权。 |

这两个文件汇总的是**每折各自选择最佳轮次**后的验证结果；它们不包含官方测试集成绩，也不代表固定同一轮次训练的五折均值。逐轮指标在各折的 `history.csv` 中。

Accuracy 是正确预测数除以验证样本数；Macro-F1 是六类 F1 的算术平均。指标均来自**各折验证集**，且每折权重依据该折验证 Macro-F1 选择，因此这些数值不等同于独立官方测试集性能。

## 6. 已完成的五折结果

以下数值来自 `results/cnn1d/fold_metrics.csv` 和 `cv_summary.csv`；均为验证集指标。

| 折 | 训练样本 | 验证样本 | 最佳轮次 | 实际轮数 | Accuracy | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,932 | 1,420 | 3 | 10 | 92.39% | 92.46% |
| 2 | 5,932 | 1,420 | 2 | 9 | 90.92% | 91.36% |
| 3 | 5,935 | 1,417 | 21 | 28 | 99.79% | 99.80% |
| 4 | 5,942 | 1,410 | 4 | 11 | 96.45% | 96.34% |
| 5 | 5,667 | 1,685 | 3 | 10 | 91.93% | 92.73% |
| **五折均值** | — | — | — | — | **94.30%** | **94.54%** |
| **五折标准差** | — | — | — | — | **3.72 个百分点** | **3.49 个百分点** |

第 3 折明显高于其他折，因此汇报时应同时呈现逐折结果和均值。五折结果仅用于模型选择和验证表现分析；独立官方测试结果见第 8 节。

## 7. 最终模型的预先确定配置

官方测试集评估固定使用本目录当前的 `SmallCNN1D` 三层卷积结构、9 通道原始信号、交叉熵损失、Adam 优化器（学习率 `0.001`）、batch size `128` 和基础随机种子 `42`。最终模型用全部 **7,352** 条官方训练样本从头训练 **3 轮**；官方测试集不参与轮数或其他参数选择。

轮数依据五折 `history.csv` 选择：五折都至少训练到第 9 轮，因此比较共同可用的第 1–9 轮。在相同轮次汇总五折验证 Macro-F1 时，第 3 轮均值最高，为 **93.16%**（第 2 轮为 92.57%，第 4 轮为 92.78%）。这是**固定第 3 轮**的五折均值；上文 **94.54%** 是各折分别挑选最佳轮次后的均值，两者的计算方法不同。

`train.py` 保持为五折验证脚本；最终训练和官方测试评估由独立的 `train_final_evaluate.py` 完成。

## 8. 最终训练与官方测试结果

在项目根目录运行：

```powershell
python src/cnn1d/train_final_evaluate.py
```

该脚本固定训练 3 轮，不执行早停。训练完成后才对 2,947 条官方测试样本进行一次推理。

| 指标 | 官方测试集结果 |
| --- | ---: |
| Accuracy | **91.48%** |
| Macro Precision | 91.56% |
| Macro Recall | 91.66% |
| Macro-F1 | **91.58%** |

各类召回率：步行 100.00%、上楼 92.36%、下楼 99.29%、坐 76.58%、站立 81.77%、躺 100.00%。主要混淆发生在坐与站之间：105 条坐被预测为站立，97 条站立被预测为坐。

所有最终产物位于 `results/cnn1d/official_test/`：

| 文件 | 内容 |
| --- | --- |
| `model.pt` | 用全部官方训练集训练 3 轮后的模型权重。 |
| `train_history.csv`、`training_curve.png` | 三轮训练损失及其曲线；无验证或测试逐轮曲线。 |
| `metrics.json` | 官方测试集总体指标、样本数和设备信息；不含训练耗时。 |
| `class_metrics.csv` | 六类的 Precision、Recall、F1 和样本数。 |
| `confusion_matrix.csv` | 带活动名称行列标题的 6×6 官方测试混淆矩阵。 |
| `predictions.csv` | 每条官方测试样本的行号、参与者编号、真实标签和预测标签；标签为 `0–5`。 |
| `run_config.json` | 预先确定的超参数、随机种子、数据来源、五折文件校验值及运行环境。 |

五折平均指标与官方测试指标来自不同评估方式，应分别报告。官方测试结果不用于继续调整上述模型配置。
