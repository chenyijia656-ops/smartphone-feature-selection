# Smartphone Feature Selection for Human Activity Recognition 基于智能手机传感器数据的人类活动识别与特征选择研究


## 项目简介（Project Overview）

本项目研究基于智能手机传感器数据的人类活动识别（Human Activity Recognition, HAR），重点关注特征选择对分类性能、计算效率和跨用户泛化能力的影响。

项目采用 UCI Human Activity Recognition Using Smartphones Dataset，使用数据集提供的561维工程特征，对六类日常活动进行分类，包括走路、上楼、下楼、坐、站立和躺下。

传统机器学习模型能够有效完成 HAR 分类，但输入特征可能存在冗余。减少特征数量有望降低计算开销，但也可能影响识别性能。

因此，本项目的核心目标是研究：能否通过可学习特征选择减少输入维度，同时维持较好的分类性能并改善计算效率？

项目将比较传统机器学习模型、互信息特征筛选、可学习特征门控和轻量级 Feature-token Transformer，通过系统实验验证不同方法的实际效果。

## 研究问题（Research Questions）

### RQ1：不同模型的人体活动识别性能如何？

使用以下模型建立人类活动分类模型，比较它们在完整561维特征下的识别性能：

- Support Vector Machine（SVM）

- XGBoost

- Random Forest

- Lightweight Feature-token Transformer

### RQ2：特征选择能否在减少特征数量的同时维持分类性能？

本项目重点比较两种方法：

Mutual Information（MI）：根据特征与活动标签之间的统计依赖程度进行排序。

Learnable Feature Gating：通过模型训练学习特征权重，并利用稀疏正则化鼓励减少冗余特征。

研究不同特征数量下的分类表现，分析特征压缩与分类性能之间的关系。

### RQ3：特征压缩能否改善计算效率？

比较不同特征数量下的推理时间、特征选择耗时、模型文件大小和神经网络参数量。

### RQ4：选择的特征是否稳定，能否推广到不同用户？

分析不同训练样本或用户组合下特征选择结果的一致性，并探索模型对未参与训练用户的泛化能力。

## 数据集介绍（Dataset）
### 1. 数据来源

数据集：UCI Human Activity Recognition Using Smartphones Dataset

数据来源：UCI Machine Learning Repository

参与者：30人

活动类别：6类

工程特征：561维

传感器：加速度计和陀螺仪

数据划分: 官方训练集与测试集

数据集链接：

https://archive-beta.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones

### 2. 活动类别
| 标签 | 活动 |
| ---- | ---- |
| Walking | 走路 |
| Walking Upstairs | 上楼 |
| Walking Downstairs | 下楼 |
| Sitting | 坐 |
| Standing | 站立 |
| Laying | 躺 |
### 3. 数据预处理

本项目主要使用数据集提供的561维工程特征，不在主实验中重新提取原始惯性信号特征。

预处理步骤包括：

1. 读取训练集、测试集、活动标签和参与者编号

2. 检查数据维度、缺失值以及不同活动类别的样本分布

3. 理解561个工程特征的组成及其传感器来源

4. 根据模型需求进行必要的数据标准化，标准化参数仅从训练数据中学习

5. 在训练集内部进行模型验证、特征选择和超参数调优

6. 使用官方测试集对最终确定的模型进行性能评估

数据泄漏控制：

为保证实验结果的可靠性，官方测试集不参与特征筛选、标准化参数拟合或模型调参。

跨用户泛化实验应根据参与者编号进行合理分组，确保验证数据中的用户不出现在对应的训练数据中。

## 研究方法（Methodology）
### 1. 全特征基线（Baseline）

使用全部561个特征训练以下模型：

SVM：高维分类基线。

XGBoost：梯度提升树模型。

Random Forest：经典树模型参照。

Lightweight Feature-token Transformer：主要深度学习研究方向。

记录 Accuracy、Macro-F1、混淆矩阵和计算效率。通过比较不同模型的分类性能和计算成本，分析完整工程特征对人体活动识别任务的有效性。

### 2. 互信息特征选择（Mutual Information）

互信息（Mutual Information, MI）用于衡量特征与活动标签之间的统计依赖程度。

本项目在训练数据上计算各特征的互信息得分，并根据得分排序，构建不同规模的特征子集。

计划比较以下特征数量：

| 特征子集 | 特征数量 |
| Top 30 | 30 |
| Top 60 | 60 |
| Top 100 |	100 |
|Top 200 | 200 |
|Full Features | 561 |

通过比较不同特征数量下的分类结果，评估传统统计特征筛选方法的效果。

### 3. 可学习特征门控（Learnable Feature Gating）

该方法为输入特征设置可学习的门控权重，使模型能够在训练过程中学习不同特征的重要程度，并通过稀疏正则化鼓励减少对冗余特征的依赖。

基本流程：

为每个特征学习门控权重

↓

使用门控权重调整输入特征

↓

通过稀疏正则化鼓励部分特征权重趋近于零

↓

根据学习到的权重对特征排序

↓

选取不同数量的特征子集

↓

重新训练分类模型并评估性能

通过与 Mutual Information 方法比较，分析可学习特征选择是否能够发现更有效的特征组合。

( 本研究不预设该方法一定优于传统统计筛选方法，最终结论以实验结果为依据。)

### 4. 轻量级 Feature-token Transformer

构建轻量级 Transformer，研究不同工程特征之间的交互关系。

初步模型结构：

输入特征 → 特征门控 → 特征嵌入 → Transformer Encoder → 分类层 → 六类活动预测

为控制计算成本和过拟合风险，模型计划采用：

较少的 Transformer Encoder 层数

较小的隐藏层维度

Dropout 正则化

轻量化模型结构

需要注意的是，UCI HAR 数据集提供的561维输入为已经提取的工程特征，而非原始时间序列。因此，本项目的 Transformer 主要用于研究不同工程特征之间的交互关系，而不是直接对原始传感器时间序列建模。

### 5. 特征解释与稳定性分析

分析不同方法选出的特征及其一致性，重点关注：

走路、上楼和下楼之间的混淆。

坐与站立之间的混淆。

特征数量减少后，各类别召回率的变化。

不同训练样本下特征排名的稳定性。

特征压缩对跨用户泛化的影响。

## 实验设计（Experiments）
| 实验 | 方法 |	研究目的 |
| ---- | ---- | ---- |
| 实验一 | SVM、XGBoost、Random Forest | 建立全特征基线 |
| 实验二 | Mutual Information | 评估传统特征筛选效果 |
| 实验三 | Learnable Feature Gating | 评估可学习特征选择 |
| 实验四 | Lightweight Transformer |	研究特征交互与压缩的关系 |
| 实验五 | 特征稳定性与跨用户分析 | 研究稳定性和泛化能力 |

所有主要对比实验应尽可能使用相同的数据划分、特征数量和评价指标。


## 评价指标（Evaluation Metrics）
### 1. 分类性能

使用以下指标评价模型的人类活动识别能力：

Accuracy（准确率）

Macro Precision（宏平均精确率）

Macro Recall（宏平均召回率）

Macro F1-score（宏平均F1分数）

Per-class Recall（各类别召回率）

Confusion Matrix（混淆矩阵）

Confusion Matrix

重点关注 Macro-F1 和各类别召回率，避免仅根据整体准确率判断模型表现。

### 2. 计算效率

保留特征数量

特征减少比例

特征选择耗时

模型训练时间

CPU 推理延迟

模型文件大小

神经网络参数量

所有计算效率实验应尽可能在一致的硬件和软件环境下进行，并明确计时是否包含特征计算、特征选择及其他数据处理步骤。需要区分输入维度减少带来的计算开销变化与实际端到端推理效率提升，避免仅根据特征数量判断模型是否更加高效。

### 3. 候选实验目标

初步目标为：

特征数量减少至少50%。

Accuracy 下降不超过1.5个百分点。

Macro-F1 下降不超过2个百分点。

以上是计划验证的目标，需要在正式实验前与指导老师确认。最终结论以真实实验结果为准。

## 预期研究贡献（Expected Contributions）

1. 特征选择方法比较

在统一实验设置下比较 Mutual Information 与 Learnable Feature Gating，分析传统统计特征选择和可学习特征选择之间的差异。

2. 特征压缩与计算效率分析

研究减少输入特征是否能够在维持分类性能的同时降低模型计算成本，并分析不同特征数量下性能与效率之间的关系。

3. 特征稳定性与跨用户泛化分析

探索不同训练数据和用户组合下特征选择结果的一致性，研究特征压缩对跨用户泛化能力的影响。

4. 类别级分类性能分析

分析不同特征选择方法对于各类人体活动识别结果的影响，特别关注容易产生混淆的活动类别。

这些属于拟验证的研究贡献，最终创新性需要通过文献对比和实验结果确定。

## 项目结构（Project Structure）
smartphone-feature-selection
│
├── README.md                     # 项目介绍与研究说明
├── LICENSE                       # 开源许可证
├── requirements.txt              # Python环境依赖
│
├── data/
│   ├── raw/                      # UCI HAR原始发布数据集
│   └── processed/                # 实验中生成的数据或特征子集（按需使用）
│
├── notebooks/                    # Jupyter Notebook实验过程
│   │
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_mutual_information.ipynb
│   ├── 04_feature_gating.ipynb
│   ├── 05_feature_transformer.ipynb
│   └── 06_evaluation_analysis.ipynb
│
├── src/                          # 可复用的Python核心代码
│   │
│   ├── data_loader.py            # 数据读取与检查
│   ├── baseline_models.py        # 传统分类模型
│   ├── feature_selection.py      # 特征选择方法
│   ├── feature_gating.py         # 可学习特征门控
│   ├── feature_transformer.py    # 轻量级Transformer
│   └── evaluation.py             # 性能与效率评价
│
├── results/
│   ├── figures/                  # 实验可视化图片
│   │   ├── activity_distribution.png
│   │   ├── confusion_matrix.png
│   │   └── feature_comparison.png
│   │
│   └── model_results.csv         # 实验性能对比结果
│
├── models/                       # 训练完成的模型文件
│
└── report/
    ├── paper.pdf                 
    └── presentation.pptx         

##  参考文献

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN.

An ensemble maximal feature subset selection for smartphone based human activity recognition. (2024). Journal of Network and Computer Applications, 226, 103875. https://doi.org/10.1016/j.jnca.2024.103875

A benchmark for domain adaptation and generalization in smartphone-based human activity recognition. (2024). Scientific Data. https://doi.org/10.1038/s41597-024-03951-4

Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.