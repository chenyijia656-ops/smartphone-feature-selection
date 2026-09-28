# Smartphone Feature Selection for Human Activity Recognition 基于智能手机传感器数据的人类活动识别与特征选择研究

## 项目简介
本项目围绕基于智能手机传感器数据的人类活动识别（Human Activity Recognition, HAR）问题展开研究。

随着智能手机中加速度计（Accelerometer）、陀螺仪（Gyroscope）等传感器的发展，手机能够持续采集人体运动过程中的传感器信号。这些数据可以用于识别用户当前进行的活动，例如走路、上下楼梯、坐、站立和躺下等。

本项目旨在利用智能手机传感器数据，通过数据挖掘和机器学习方法建立人类活动识别模型，研究不同传感器特征对于活动分类的影响。

同时，本项目重点关注特征选择（Feature Selection）问题，探索在减少特征数量、降低模型复杂度的情况下，是否仍然能够保持较好的分类性能。

## 研究问题
本项目主要围绕以下两个研究问题展开：

### 研究问题1：智能手机传感器特征能否准确识别人类活动？

智能手机中的加速度计（Accelerometer）和陀螺仪（Gyroscope）能够记录人体运动过程中的变化信息。本研究将利用这些传感器提取的特征，建立机器学习分类模型，判断模型是否能够有效区分不同的人类活动。

本项目关注的六类活动包括：

- Walking（走路）
- Walking Upstairs（上楼）
- Walking Downstairs（下楼）
- Sitting（坐）
- Standing（站立）
- Laying（躺）


### 研究问题2：减少特征数量后，模型性能是否仍然可以接受？

原始数据集包含561个传感器相关特征。虽然更多特征可能包含更多信息，但也可能增加模型训练时间和复杂度。

因此，本研究将使用特征选择方法筛选重要特征，并比较：

- 使用全部561个特征时的模型性能；
- 使用减少后的特征子集时的模型性能。

通过实验分析特征数量减少对于分类准确率和模型性能的影响。

## 数据集介绍
本项目使用的数据集为 **Human Activity Recognition Using Smartphones Dataset（基于智能手机的人类活动识别数据集）**。

该数据集由 UCI Machine Learning Repository 提供，是人类活动识别领域中广泛使用的公开数据集。


### 数据来源

数据集链接：

https://archive-beta.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones


### 数据集特点

该数据集通过智能手机内置传感器采集人体运动数据，主要包括：

- 加速度计（Accelerometer）
- 陀螺仪（Gyroscope）


实验参与者在进行不同活动时，手机传感器会记录人体运动产生的信号，并进一步提取用于机器学习分析的特征。


### 数据规模

数据集包含：

- 30名参与者（Subjects）
- 6种人体活动类别
- 561个传感器提取特征


六种活动类别包括：

| 标签 | 活动 |
| ---- | ---- |
| Walking | 走路 |
| Walking Upstairs | 上楼 |
| Walking Downstairs | 下楼 |
| Sitting | 坐 |
| Standing | 站立 |
| Laying | 躺 |


### 数据划分

原始数据已经划分为训练集（Training Set）和测试集（Test Set）。

本项目将在训练集上建立机器学习模型，并使用测试集评估模型对未知活动数据的识别能力。

## 研究方法与技术路线

本项目基于 UCI Human Activity Recognition Using Smartphones Dataset 展开研究。由于该数据集已经完成传感器信号处理、特征提取以及训练集和测试集划分，因此本项目不进行原始数据预处理，而重点关注特征分析、特征选择以及模型性能比较。

整体研究流程如下：
传感器数据

↓

数据理解与质量检查

↓

全特征Baseline模型

↓

特征选择

↓

不同特征数量与传感器组合的对比实验

↓

性能、效率、可解释性的综合结论

---

### 1. 数据理解与质量检查（Data Understanding）

首先对已有数据集进行探索分析，了解数据结构和特征特点，包括：

- 查看训练集和测试集规模；
- 分析561个传感器特征的组成；
- 查看不同活动类别的数据分布；
- 检查数据完整性；
- 分析加速度计和陀螺仪相关特征。


通过数据理解阶段，为后续模型建立和特征选择提供基础。

### 2. 全特征Baseline模型（Baseline Model）
首先使用数据集提供的全部561个特征建立基础分类模型。

该实验作为性能基准，作为后续特征选择实验的性能参考：
- 智能手机传感器特征整体是否能够有效区分不同人体活动；
- 不同机器学习模型在完整特征空间下的分类能力。

### 3. 特征选择（Feature Selection）

针对数据集中大量传感器特征可能存在的信息冗余问题，采用特征选择方法筛选重要特征，并分析不同特征对于活动识别任务的贡献。
主要方法包括：

SelectKBest
根据统计指标评价每个特征与活动类别之间的相关性，并选择排名靠前的特征。

Random Forest Feature Importance
利用随机森林模型计算不同特征的重要程度，分析哪些传感器特征对于活动识别贡献较大。

Recursive Feature Elimination (RFE)
通过递归删除不重要特征，寻找能够保持较好分类性能的特征组合。

### 4. 对比实验与结果分析（Comparison & Analysis）

通过比较不同特征组合下的模型表现，分析特征减少对于分类性能、模型效率以及结果解释性的影响。


## 项目结构

本项目采用常规数据挖掘项目结构，将数据、代码、实验过程、模型结果和最终报告分开管理，便于后续开发、复现实验和展示研究过程。

```text
smartphone-feature-selection
│
├── README.md                 # 项目介绍与研究说明
├── LICENSE                   # 开源许可证
├── requirements.txt          # Python环境依赖
│
├── data/
│   ├── raw/                  # 原始数据集
│   └── processed/            # 清洗和预处理后的数据
│
├── notebooks/                # Jupyter Notebook实验分析过程
│   │
│   ├── 01_data_understanding.ipynb
|   ├── 02_baseline_model.ipynb
|   ├── 03_feature_analysis.ipynb
|   ├── 04_feature_selection.ipynb
|   ├── 05_model_comparison.ipynb
|   ├── 06_results_visualization.ipynb
│
├── src/                      # 可复用的核心Python代码
│   │
│   ├── data_loader.py        # 数据读取
│   ├── preprocessing.py      # 数据预处理
│   ├── feature_selection.py  # 特征选择方法
│   ├── models.py             # 机器学习模型训练
│   └── evaluation.py         # 模型评价指标
│
├── results/
│   │
│   ├── figures/              # 实验可视化结果
│   │   ├── activity_distribution.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   │
│   └── model_results.csv     # 模型性能比较结果
│
├── models/
│   └── best_model.pkl        # 保存表现最好的模型
│
└── report/
    ├── paper.pdf             # 最终项目报告
    └── presentation.pptx     
```

其中，`notebooks/` 主要用于记录完整的数据挖掘实验过程，适合展示从数据探索到模型评价的研究步骤；`src/` 用于存放可以重复调用的 Python 代码，使项目结构更加清晰、规范。

`results/` 和 `models/` 用于保存实验输出，包括模型性能结果、可视化图片和训练完成的模型；`report/` 用于保存最终报告和展示材料。


## 实验设计
本项目将通过多个实验验证研究问题，并比较不同特征数量和机器学习模型对于人类活动识别任务的影响。


## 实验1：Baseline实验

首先使用数据集提供的全部561个特征进行模型训练。

实验流程：
561个特征

↓

机器学习模型训练

↓

测试集预测

↓

性能评价

该实验作为基础模型（Baseline），用于评估完整特征情况下模型的分类能力。


---

## 实验2：特征选择与特征组合优化实验

在完整特征实验基础上，使用特征选择方法减少特征数量。

实验流程：
561个特征

↓

特征重要性分析

↓

选择重要特征子集

↓

模型训练

↓

性能比较

将比较不同数量特征下的模型效果，例如：

- 完整561个特征；
- 特征选择后的重要特征子集；
- 不同数量特征组合下的模型性能

通过实验分析特征减少对于活动识别准确率、模型效率以及特征解释性的影响。

---

## 实验3：不同机器学习模型比较

本项目将比较不同机器学习算法在人类活动识别任务中的表现。

比较模型包括：

| 模型 | 说明 |
| ---- | ---- |
| SVM | 适用于高维分类问题的经典算法 |
| Random Forest | 可以进行分类并提供特征重要性 |
| XGBoost | 基于梯度提升的集成学习算法 |


---

## 实验评价

不同实验结果将通过以下指标进行比较：

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-score
- Confusion Matrix（混淆矩阵）


最终分析：

1. 智能手机传感器特征是否能够有效区分不同人体活动；
2. 特征数量减少后模型性能是否仍然保持在可接受范围；
3. 哪些特征对于活动识别贡献最大。

## 参考文献
1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).

A Public Domain Dataset for Human Activity Recognition Using Smartphones.

ESANN.

2. Kumar et al. (2024).

An ensemble maximal feature subset selection for smartphone based human activity recognition.

Journal of Network and Computer Applications, 226, 103875.