"""第51步：分析GRRF五折按用户分组验证（不重新训练，也不使用官方测试集）。
请从项目根目录执行：.\\.venv\\Scripts\\python.exe .\\src\\04_analyze_grrf_cv5.py
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

root = Path.cwd()
out = root / "results"
metrics_path = out / "grrf_cv5_fold_metrics.csv"
features_path = out / "grrf_cv5_selected_features.csv"
times_path = out / "grrf_cv5_selection_times.csv"
for path in [metrics_path, features_path, times_path]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}；请在项目根目录运行，并确认第50步已完成。")

m = pd.read_csv(metrics_path)
s = pd.read_csv(features_path)
t = pd.read_csv(times_path)
required_m = {"fold", "method", "n_features", "accuracy", "macro_f1", "training_seconds", "prediction_seconds"}
required_s = {"fold", "gamma_grrf", "feature_index_python"}
required_t = {"fold", "gamma_grrf", "selection_total_seconds"}
assert required_m <= set(m.columns), f"指标文件缺列: {required_m - set(m.columns)}"
assert required_s <= set(s.columns), f"特征文件缺列: {required_s - set(s.columns)}"
assert required_t <= set(t.columns), f"计时文件缺列: {required_t - set(t.columns)}"
assert set(m.fold.unique()) == {1, 2, 3, 4, 5}, "折编号应为1~5"
assert not m.duplicated(["fold", "method"]).any(), "每折每方法应只有一条结果"

# 同一折的配对比较；每折看是否满足团队制定的候选目标
baseline = m.loc[m.method == "Full_561", ["fold", "accuracy", "macro_f1"]].rename(
    columns={"accuracy": "full_accuracy", "macro_f1": "full_macro_f1"}
)
assert len(baseline) == 5
paired = m.merge(baseline, on="fold", validate="many_to_one")
paired["accuracy_drop_pp"] = (paired.full_accuracy - paired.accuracy) * 100
paired["macro_f1_drop_pp"] = (paired.full_macro_f1 - paired.macro_f1) * 100
paired["reduction_percent"] = (1 - paired.n_features / 561) * 100
paired["meets_all_targets"] = (
    (paired.reduction_percent >= 50)
    & (paired.accuracy_drop_pp <= 1.5)
    & (paired.macro_f1_drop_pp <= 2.0)
)
paired.loc[paired.method == "Full_561", "meets_all_targets"] = False
paired.to_csv(out / "grrf_cv5_paired_differences.csv", index=False, encoding="utf-8-sig")

print("\n===== 五折逐折性能及与全特征比较 =====")
print(paired[["fold", "method", "n_features", "accuracy", "macro_f1", "accuracy_drop_pp", "macro_f1_drop_pp"]].round(4).to_string(index=False))
print("\n===== 每组达到三项目标的折数（仅作描述） =====")
print(paired[paired.method != "Full_561"].groupby("method").meets_all_targets.agg(["sum", "count"]).to_string())

# 交叉验证的特征稳定性：每个gamma每折的选择集合，以及10对折之间的Jaccard
stability = []
frequency = []
for gamma in sorted(s.gamma_grrf.unique()):
    sg = s[np.isclose(s.gamma_grrf, gamma)]
    fold_sets = {}
    for fold in range(1, 6):
        arr = sg.loc[sg.fold == fold, "feature_index_python"].astype(int).to_numpy()
        assert len(arr) > 0 and len(np.unique(arr)) == len(arr), f"gamma={gamma}, fold={fold} 的特征为空或有重复"
        assert ((arr >= 0) & (arr < 561)).all()
        fold_sets[fold] = set(arr.tolist())
    for (fa, a), (fb, b) in combinations(fold_sets.items(), 2):
        stability.append({"gamma_grrf": gamma, "fold_a": fa, "fold_b": fb,
                          "n_a": len(a), "n_b": len(b), "intersection": len(a & b),
                          "union": len(a | b), "jaccard": len(a & b) / len(a | b)})
    for feature in sorted(set.union(*fold_sets.values())):
        count = sum(feature in vals for vals in fold_sets.values())
        frequency.append({"gamma_grrf": gamma, "feature_index_python": feature, "selected_in_n_folds": count,
                          "selection_rate": count / 5})

stable_df = pd.DataFrame(stability)
freq_df = pd.DataFrame(frequency)
stable_df.to_csv(out / "grrf_cv5_pairwise_jaccard.csv", index=False, encoding="utf-8-sig")
freq_df.to_csv(out / "grrf_cv5_feature_frequency.csv", index=False, encoding="utf-8-sig")
print("\n===== 5折之间的特征集合重叠（每个gamma共10对） =====")
print(stable_df.groupby("gamma_grrf").jaccard.agg(["mean", "std", "min", "max"]).round(4).to_string())
print("\n===== 每个gamma出现于全部5折的特征数量 =====")
print(freq_df[freq_df.selected_in_n_folds == 5].groupby("gamma_grrf").size().reindex(sorted(s.gamma_grrf.unique()), fill_value=0).to_string())

# 总耗时分开展示：GRRF特征选择耗时 + SVM训练 + SVM预测
assert not t.duplicated(["fold", "gamma_grrf"]).any(), "每折每个gamma应有一条GRRF耗时"
selected_methods = paired[paired.method != "Full_561"].copy()
selected_methods["gamma_grrf"] = selected_methods.method.str.replace("GRRF_gamma_", "", regex=False).astype(float)
timed = selected_methods.merge(t[["fold", "gamma_grrf", "selection_total_seconds"]],
                               on=["fold", "gamma_grrf"], validate="one_to_one")
assert len(timed) == len(selected_methods), "没有找到所有GRRF实验的特征选择耗时"
timed["total_seconds"] = timed.selection_total_seconds + timed.training_seconds + timed.prediction_seconds
timed.to_csv(out / "grrf_cv5_pipeline_times.csv", index=False, encoding="utf-8-sig")
print("\n===== 各gamma平均计时（秒） =====")
print(timed.groupby("method")[["selection_total_seconds", "training_seconds", "prediction_seconds", "total_seconds"]].mean().round(3).to_string())
print("\n注意：总耗时是此实验的选择+训练+预测，不是手机原始传感器到预测的端到端耗时。")
print("\n生成文件：grrf_cv5_paired_differences.csv、grrf_cv5_pairwise_jaccard.csv、")
print("grrf_cv5_feature_frequency.csv、grrf_cv5_pipeline_times.csv")
