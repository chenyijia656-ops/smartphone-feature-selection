"""Evaluate full-training GRRF feature sets once on the official test set."""
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"
OUT = ROOT / "results" / "official_test_grrf"
features = pd.read_csv(OUT / "selected_features.csv")
times = pd.read_csv(OUT / "selection_times.csv")
expected = ["GRRF_gamma025_nt100_m23", "GRRF_gamma1_nt500_m23"]
assert set(features.method) == set(expected) and set(times.method) == set(expected)
sets = {"Full_561": np.arange(561)}
for method in expected:
    cols = features.loc[features.method == method, "feature_index_python"].to_numpy(dtype=int)
    assert len(cols) == len(np.unique(cols)) and np.all((cols >= 0) & (cols < 561))
    assert len(cols) == int(times.loc[times.method == method, "n_features"].iloc[0])
    sets[method] = cols

Xtr = pd.read_csv(DATA / "train" / "X_train.txt", sep=r"\s+", header=None).to_numpy()
ytr = np.loadtxt(DATA / "train" / "y_train.txt", dtype=int)
Xte = pd.read_csv(DATA / "test" / "X_test.txt", sep=r"\s+", header=None).to_numpy()
yte = np.loadtxt(DATA / "test" / "y_test.txt", dtype=int)
assert Xtr.shape == (7352, 561) and Xte.shape == (2947, 561)
names = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
summary, class_rows = [], []
for method, cols in sets.items():
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    start = perf_counter()
    model.fit(Xtr[:, cols], ytr)
    train_seconds = perf_counter() - start
    start = perf_counter()
    pred = model.predict(Xte[:, cols])
    predict_seconds = perf_counter() - start
    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, labels=np.arange(1, 7), average="macro")
    selection_seconds = 0 if method == "Full_561" else float(times.loc[times.method == method, "selection_seconds"].iloc[0])
    summary.append(dict(method=method, n_features=len(cols), feature_reduction_percent=(1-len(cols)/561)*100,
                        train_samples=len(ytr), test_samples=len(yte), accuracy=acc, macro_f1=mf1,
                        svm_C=1.0, svm_gamma="scale", scaler="StandardScaler",
                        selection_seconds=selection_seconds, training_seconds=train_seconds,
                        test_prediction_seconds=predict_seconds))
    precision, recall, f1, support = precision_recall_fscore_support(yte, pred, labels=np.arange(1, 7))
    class_rows.extend(dict(method=method, activity=name, precision=float(precision[i]),
                           recall=float(recall[i]), f1=float(f1[i]), support=int(support[i]))
                      for i, name in enumerate(names))
    cm = confusion_matrix(yte, pred, labels=np.arange(1, 7))
    pd.DataFrame(cm, index=names, columns=names).to_csv(OUT / f"confusion_{method}.csv", encoding="utf-8-sig")

table = pd.DataFrame(summary)
table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(class_rows).to_csv(OUT / "class_report.csv", index=False, encoding="utf-8-sig")
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})
navy, teal, orange, muted, grid = "#183153", "#278B8A", "#D8833B", "#607080", "#E6ECF1"
titles = ["全特征基线", "GRRF γ=0.25", "GRRF γ=1"]
zh_names = ["步行", "上楼", "下楼", "坐", "站立", "躺下"]
palette = [navy, teal, orange]

# Slide 1: model performance with the feature count on the same line.
fig, ax = plt.subplots(figsize=(13.2, 7.4))
fig.subplots_adjust(left=0.23, right=0.96, top=0.78, bottom=0.18)
fig.suptitle("官方测试集｜特征压缩与识别性能", x=0.06, y=0.96,
             ha="left", fontsize=23, weight="bold", color=navy)
fig.text(0.06, 0.88, "7,352 条训练样本  ·  2,947 条测试样本  ·  相同的标准化 + RBF-SVM",
         fontsize=12, color=muted)
ypos = np.arange(3)[::-1]
for i, row in table.iterrows():
    accuracy, macro_f1 = row.accuracy * 100, row.macro_f1 * 100
    ax.plot([0, accuracy], [ypos[i], ypos[i]], color=grid, lw=15,
            solid_capstyle="round", zorder=1)
    ax.plot([0, accuracy], [ypos[i], ypos[i]], color=palette[i], lw=9,
            solid_capstyle="round", zorder=2)
    ax.scatter([macro_f1], [ypos[i]], s=110, color="white", edgecolor=palette[i],
               lw=2.6, zorder=3)
    ax.text(accuracy + 1.3, ypos[i] + 0.15, f"{accuracy:.2f}%", color=palette[i],
            fontsize=17, weight="bold", va="center")
    ax.text(accuracy + 1.3, ypos[i] - 0.19, f"Macro-F1  {macro_f1:.2f}%",
            color=muted, fontsize=10, va="center")
ax.set_yticks(ypos, [f"{titles[i]}\n{int(table.n_features.iloc[i])} 个特征" for i in range(3)])
ax.tick_params(axis="y", length=0, pad=15, labelsize=13)
ax.set_xlim(0, 112)
ax.set_ylim(-0.55, 2.55)
ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_xlabel("准确率（彩色横线）  ·  Macro-F1（空心圆）", color=muted, labelpad=12)
ax.grid(axis="x", color=grid, zorder=0)
ax.spines[["left", "bottom"]].set_visible(False)
fig.text(0.06, 0.06,
         f"24 个特征压缩 {table.feature_reduction_percent.iloc[1]:.2f}%，准确率较基线下降 {(table.accuracy.iloc[0]-table.accuracy.iloc[1])*100:.2f} 个百分点。",
         fontsize=12, color=navy, weight="bold")
fig.savefig(OUT / "performance_comparison.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)

# Slide 2: row-normalized confusion matrices, with counts retained in brackets.
fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
fig.subplots_adjust(left=0.07, right=0.98, top=0.75, bottom=0.18, wspace=0.22)
fig.suptitle("官方测试集｜六类活动识别与混淆", x=0.04, y=0.96,
             ha="left", fontsize=23, weight="bold", color=navy)
fig.text(0.04, 0.88, "每行是实际活动；单元格显示该类样本中被预测为对应活动的比例，括号内为样本数。",
         fontsize=11, color=muted)
cmap = colors.LinearSegmentedColormap.from_list("project_blues", ["#F4F7FA", "#A9C5DF", "#3572AB", navy])
for i, method in enumerate(sets):
    ax = axes[i]
    cm = pd.read_csv(OUT / f"confusion_{method}.csv", index_col=0).loc[names, names].to_numpy()
    percentages = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(percentages, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    for r in range(6):
        for c in range(6):
            val = percentages[r, c]
            shown_percent = "<1%" if 0 < val < 0.5 else f"{val:.0f}%"
            label = "·" if cm[r, c] == 0 else f"{shown_percent}\n({cm[r, c]})"
            ax.text(c, r, label, ha="center", va="center", fontsize=9,
                    color="white" if val >= 55 else navy,
                    weight="bold" if r == c else "normal")
    ax.set_xticks(range(6), zh_names, fontsize=11)
    ax.set_yticks(range(6), zh_names if i == 0 else [""] * 6, fontsize=11)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"{titles[i]}  ·  {int(table.n_features.iloc[i])} 特征\n准确率 {table.accuracy.iloc[i]:.2%}",
                 color=palette[i], fontsize=13, weight="bold", pad=17)
    ax.set_xlabel("预测活动", color=muted, labelpad=10)
    if i == 0:
        ax.set_ylabel("实际活动", color=muted, labelpad=8)
fig.text(0.04, 0.06, "阅读提示：对角线越深越好；非对角线数字越大，代表两类活动越容易被混淆。",
         fontsize=11, color=muted)
fig.savefig(OUT / "confusion_matrices.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)

# Slide 3: activity recall, so the cost of compression is visible by class.
recall = pd.DataFrame(class_rows).pivot(index="activity", columns="method", values="recall").loc[names, list(sets)] * 100
values = recall.to_numpy()
fig, ax = plt.subplots(figsize=(12.5, 7.5))
fig.subplots_adjust(left=0.14, right=0.98, top=0.77, bottom=0.16)
fig.suptitle("官方测试集｜各活动召回率", x=0.05, y=0.96,
             ha="left", fontsize=23, weight="bold", color=navy)
fig.text(0.05, 0.88, "召回率表示该活动的样本有多少被正确识别；横向比较可看到压缩后的损失。",
         fontsize=11, color=muted)
recall_cmap = colors.LinearSegmentedColormap.from_list("recall", ["#FCECE0", "#F7FAFC", "#A6CFC8", "#237E7F"])
ax.imshow(values, cmap=recall_cmap, vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(3), ["全特征\n561 个", "GRRF γ=0.25\n24 个", "GRRF γ=1\n8 个"], fontsize=12)
ax.set_yticks(range(6), zh_names, fontsize=12)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
for r in range(6):
    for c in range(3):
        val = values[r, c]
        ax.text(c, r, f"{val:.1f}%", ha="center", va="center", fontsize=16,
                color="white" if val >= 90 else navy, weight="bold")
fig.text(0.05, 0.055, "24 特征方案主要损失在“坐”与“下楼”；8 特征方案的“上楼”和“步行”下降更明显。",
         fontsize=12, color=navy, weight="bold")
fig.savefig(OUT / "activity_recall.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(table.to_string(index=False))
