
from pathlib import Path
import pandas as pd

root = Path.cwd()
results_dir = root / "results"

# 读取不同gamma下的特征集合
path = results_dir / "grrf_group_selected_features.csv"
df = pd.read_csv(path)

# 显示每个gamma选中的特征
for gamma, group in df.groupby("gamma_grrf"):
    features = sorted(
        group["feature_index_python"].astype(int).tolist()
    )
    print(f"\ngamma={gamma}")
    print(f"特征数：{len(features)}")
    print(f"Python索引：{features}")

# 比较gamma=0.75与gamma=1的特征重叠
sets = {
    float(gamma): set(
        group["feature_index_python"].astype(int)
    )
    for gamma, group in df.groupby("gamma_grrf")
}

set_075 = sets[0.75]
set_1 = sets[1.0]

intersection = set_075 & set_1
union = set_075 | set_1

print("\n===== gamma=0.75 与 gamma=1 的比较 =====")
print("gamma=0.75 特征数：", len(set_075))
print("gamma=1 特征数：", len(set_1))
print("共同特征数：", len(intersection))
print("仅 gamma=0.75 选中：", sorted(set_075 - set_1))
print("仅 gamma=1 选中：", sorted(set_1 - set_075))

jaccard = len(intersection) / len(union)
print(f"Jaccard相似度：{jaccard:.4f}")

# 输出每个gamma特征集合的重叠情况
gamma_values = sorted(sets.keys())
overlap_rows = []

for g1 in gamma_values:
    for g2 in gamma_values:
        a, b = sets[g1], sets[g2]
        overlap_rows.append({
            "gamma_1": g1,
            "gamma_2": g2,
            "shared_features": len(a & b),
            "union_features": len(a | b),
            "jaccard_similarity": len(a & b) / len(a | b)
        })

overlap_df = pd.DataFrame(overlap_rows)
output_path = results_dir / "grrf_feature_set_overlap.csv"
overlap_df.to_csv(output_path, index=False)

print("\n特征重叠矩阵已保存到：", output_path)
