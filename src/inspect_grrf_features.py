
from pathlib import Path
import pandas as pd

root = Path.cwd()
data_dir = root / "data" / "raw" / "UCI HAR Dataset"
results_dir = root / "results"

# 读取UCI HAR官方特征名称
feature_path = data_dir / "features.txt"

features = pd.read_csv(
    feature_path,
    sep=r"\s+",
    header=None,
    names=["feature_id", "feature_name"]
)

# 读取GRRF分组实验的特征集合
selected_path = (
    results_dir / "grrf_group_selected_features.csv"
)

selected = pd.read_csv(selected_path)

# Python索引从0开始，features.txt的编号从1开始
for gamma in [0.75, 1.0]:
    group = selected[
        selected["gamma_grrf"] == gamma
    ]

    feature_indices = (
        group["feature_index_python"]
        .astype(int)
        .tolist()
    )

    print(f"\n===== gamma={gamma} =====")

    for idx in feature_indices:
        row = features.iloc[idx]

        print(
            f"Python索引={idx:3d} | "
            f"UCI特征编号={int(row['feature_id']):3d} | "
            f"{row['feature_name']}"
        )

# 单独比较不同的两个特征
print("\n===== 两个不同特征的详细信息 =====")

for idx in [53, 353]:
    row = features.iloc[idx]

    print(
        f"Python索引={idx} | "
        f"UCI特征编号={int(row['feature_id'])} | "
        f"特征名称={row['feature_name']}"
    )

