
from pathlib import Path
import pandas as pd

root = Path.cwd()
results_dir = root / "results"

# 读取各类别分类报告
report = pd.read_csv(
    results_dir / "grrf_single_feature_class_report.csv"
)

# 只比较最重要的三个特征集合
methods = [
    "Common_6",
    "Common_6_plus_gamma075_feature",
    "Common_6_plus_gamma1_feature",
]

report = report[report["method"].isin(methods)].copy()

# 输出每类活动的Recall
recall_table = report.pivot(
    index="class_label",
    columns="method",
    values="recall"
)

print("\n===== 各类活动召回率 =====")
print(recall_table.round(4).to_string())

# 输出每类活动的F1
f1_table = report.pivot(
    index="class_label",
    columns="method",
    values="f1"
)

print("\n===== 各类活动F1 =====")
print(f1_table.round(4).to_string())

# 保存类别级比较结果
class_comparison = report[
    ["method", "class_label", "precision",
     "recall", "f1", "support"]
].copy()

class_comparison.to_csv(
    results_dir / "grrf_ablation_class_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# 读取混淆矩阵明细
cm = pd.read_csv(
    results_dir / "grrf_single_feature_confusion.csv"
)

cm = cm[cm["method"].isin(methods)].copy()

# 去掉正确分类的对角线，只保留误分类
errors = cm[
    cm["true_class"] != cm["predicted_class"]
].copy()

# 找出每种方法最常见的5种误分类
top_errors = (
    errors.sort_values(
        ["method", "count"],
        ascending=[True, False]
    )
    .groupby("method")
    .head(5)
)

print("\n===== 各方法最常见的误分类 =====")
print(top_errors.to_string(index=False))

top_errors.to_csv(
    results_dir / "grrf_ablation_top_confusions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n分析完成，已保存类别级对比结果。")
