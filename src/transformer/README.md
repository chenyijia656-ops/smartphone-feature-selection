# 轻量级时间序列 Transformer：模型结构

本目录使用 UCI HAR 的 9 通道、128 时间点惯性信号作为输入。`model.py` 定义网络；`train_cv.py` 复用 `src/cnn1d/data.py` 的九通道读取及 `src/cnn1d/train.py` 的用户划分校验、DataLoader、确定性设置、验证指标计算和训练曲线保存；`train_final_evaluate.py` 使用全部官方训练集训练最终模型，再评估官方测试集。

`model.py` 中的 `LightweightTimeSeriesTransformer` 依次执行：

| 步骤 | 单条样本的形状 | 设置 |
| --- | --- | --- |
| 输入 | `(9, 128)` | 9 路传感器信号 |
| 切片 | `(16, 72)` | 每 8 个时间点构成 1 个片段；每片段为 `9 × 8 = 72` 个值 |
| 片段嵌入 | `(16, 64)` | 共享线性层 `72 → 64` |
| 位置编码 | `(16, 64)` | 16 个可学习的位置向量，加到片段嵌入上 |
| Transformer Encoder | `(16, 64)` | 2 层、4 个注意力头、前馈层宽度 128、Dropout 0.1、GELU |
| 平均池化 | `(64,)` | 对 16 个片段的输出取平均 |
| 分类层 | `(6,)` | 线性层输出六个类别 logit |

模型输出是未经 softmax 的 logit，后续训练时可直接使用 `CrossEntropyLoss`，标签沿用 `src/cnn1d/data.py` 转换后的 `0–5`。输入张量应为 `float32`，形状为 `(batch, 9, 128)`。

在项目根目录进行不训练的前向形状检查：

```powershell
python src/transformer/model.py
```

## 五折验证

`train_cv.py` 读取现有的 `results/grrf_cv5_splits.csv`，在官方训练集内按参与者运行五折，不读取官方测试集。每折以验证集 Macro-F1 选择最佳权重；其余指标和混淆矩阵按 CNN 相同的计算程序生成。原 CNN 的九通道流程没有额外信号标准化，本实验沿用相同的原始 `float32` 输入和 `0–5` 标签。

默认使用 AdamW（学习率 `3e-4`、权重衰减 `0.01`）、batch size `128`、最多 `40` 轮、验证 Macro-F1 连续 `7` 轮未提高则早停。基础随机种子为 `42`，第 `k` 折使用 `42 + k`；开启 PyTorch 确定性算法。

在项目根目录运行全部五折，或只运行指定一折：

```powershell
python src/transformer/train_cv.py
python src/transformer/train_cv.py --fold 1
```

每折输出到 `results/transformer/fold_k/`：`model.pt`、`history.csv`、`training_curves.png`、`metrics.json`、`confusion_matrix.csv` 和 `run_config.json`。`metrics.json` 不含训练耗时；训练耗时只在终端输出。五折齐全时生成 `results/transformer/fold_metrics.csv` 和 `cv_summary.csv`。

### 五折汇总文件

实际文件名是 **`fold_metrics.csv`**。两个文件均位于 `results/transformer/`，数值以 `0–1` 的比例保存；下文结果表将其换算为百分比。

| 文件 | 字段与含义 |
| --- | --- |
| `fold_metrics.csv` | 每折一行。`fold` 是折号；`accuracy`、`macro_f1` 是该折**最佳模型**在对应验证集上的指标；`best_epoch` 是验证 Macro-F1 最高的轮次；`epochs_trained` 是包含早停等待在内实际运行的轮数；`train_samples`、`n_val_samples` 分别是训练和验证样本数。 |
| `cv_summary.csv` | 对 `fold_metrics.csv` 中五折的 `accuracy` 和 `macro_f1` 分别计算算术平均（`mean` 行）与样本标准差（`std` 行）。每折权重相同，不按验证样本数加权。 |

这两个文件汇总的是**每折各自选择最佳轮次**后的验证结果；它们不包含官方测试集成绩，也不代表固定同一轮次训练的五折均值。逐轮指标在各折的 `history.csv` 中。

## 已完成的五折结果

以下均为官方训练集内部、按参与者分组的验证结果；每折保存验证 Macro-F1 最高轮次的模型。

| 折 | 训练样本 | 验证样本 | 最佳轮次 | 实际轮数 | Accuracy | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,932 | 1,420 | 25 | 32 | 93.10% | 93.28% |
| 2 | 5,932 | 1,420 | 10 | 17 | 89.51% | 89.54% |
| 3 | 5,935 | 1,417 | 21 | 28 | 98.66% | 98.71% |
| 4 | 5,942 | 1,410 | 20 | 27 | 96.74% | 96.70% |
| 5 | 5,667 | 1,685 | 27 | 34 | 90.80% | 91.43% |
| **五折均值** | — | — | — | — | **93.76%** | **93.93%** |
| **五折标准差** | — | — | — | — | **3.88 个百分点** | **3.76 个百分点** |

五折指标来自 `results/transformer/fold_metrics.csv` 和 `cv_summary.csv`。五折实验仅使用官方训练集；独立官方测试结果见下一节。

## 最终模型与官方测试

最终配置在测试前根据五折逐轮记录确定。五折都训练到至少第 17 轮，因此比较共同可用的第 1–17 轮；**第 15 轮**的平均验证 Macro-F1 最高，为 **92.31%**。这与上文逐折各选最佳轮次所得的 **93.93%** 是不同口径。最终模型沿用五折时的原始九通道输入、模型结构、AdamW（学习率 `3e-4`、权重衰减 `0.01`）、batch size `128` 和确定性设置，使用随机种子 `42`，在全部 **7,352** 条官方训练样本上从头训练固定 **15 轮**，不进行早停。

在项目根目录运行：

```powershell
python src/transformer/train_final_evaluate.py
```

脚本在训练结束后才对 **2,947** 条官方测试样本推理一次。CPU 训练耗时约 **35.80 秒**，仅在终端输出，不写入 `metrics.json`。

| 指标 | 官方测试集结果 |
| --- | ---: |
| Accuracy | **89.65%** |
| Macro Precision | 89.64% |
| Macro Recall | 89.78% |
| Macro-F1 | **89.65%** |

坐与站仍是主要混淆：85 条坐被预测为站立，106 条站立被预测为坐。各类详细指标见 `results/transformer/official_test/class_metrics.csv`。

所有最终产物存放在 `results/transformer/official_test/`：

| 文件 | 内容 |
| --- | --- |
| `model.pt` | 全部官方训练集训练 15 轮后的模型权重。 |
| `train_history.csv`、`training_curve.png` | 逐轮训练损失及曲线。 |
| `metrics.json` | 官方测试集总体指标、样本数和设备信息；不含训练耗时。 |
| `class_metrics.csv` | 六类的 Precision、Recall、F1 与样本数。 |
| `confusion_matrix.csv` | 带活动名称行列标题的 6×6 混淆矩阵。 |
| `predictions.csv` | 每条测试样本的行号、参与者编号、真实与预测标签（标签为 `0–5`）。 |
| `run_config.json` | 固定超参数、轮数选择依据、划分文件校验值和运行环境。 |

官方测试集没有参与模型或轮数选择。五折验证结果与独立测试结果应分别报告。
