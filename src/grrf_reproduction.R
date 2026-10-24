
# GRRF 论文复现：基础实现检查
# 用于验证算法和特征输出，不是最终论文对照结果

library(RRF)

# 1. 设置随机种子
set.seed(42)

# 2. 定位项目数据
root <- getwd()
data_dir <- file.path(root, "data", "raw", "UCI HAR Dataset")

train_x_path <- file.path(data_dir, "train", "X_train.txt")
train_y_path <- file.path(data_dir, "train", "y_train.txt")

stopifnot(file.exists(train_x_path))
stopifnot(file.exists(train_y_path))

# 3. 读取官方训练集
X_train <- as.matrix(read.table(train_x_path, header = FALSE))
y_train <- factor(scan(train_y_path, quiet = TRUE))

stopifnot(nrow(X_train) == 7352)
stopifnot(ncol(X_train) == 561)
stopifnot(length(y_train) == 7352)

cat("训练集维度：", dim(X_train), "\n")

# 4. 设置初步实验参数
ntree <- 100
mtry <- floor(sqrt(ncol(X_train)))
gamma_grrf <- 1

# 5. 训练普通 RF，得到 Gini 特征重要性
rf <- RRF(
  x = X_train,
  y = y_train,
  ntree = ntree,
  mtry = mtry,
  flagReg = 0,
  importance = TRUE
)

importance_rf <- rf$importance[, "MeanDecreaseGini"]
importance_rf[!is.finite(importance_rf)] <- 0

stopifnot(max(importance_rf) > 0)

# 6. 计算 GRRF 的引导式正则化系数
normalized_importance <- importance_rf / max(importance_rf)

coef_reg <- (1 - gamma_grrf) +
  gamma_grrf * normalized_importance

# 7. 训练 GRRF
grrf <- RRF(
  x = X_train,
  y = y_train,
  ntree = ntree,
  mtry = mtry,
  coefReg = coef_reg,
  flagReg = 1
)


# 诊断 GRRF 为什么只选出少量特征

cat("\n===== GRRF 中间结果诊断 =====\n")

# 1. 检查普通随机森林的重要性分数
cat("\n【1】RF 特征重要性\n")
cat("特征总数：", length(importance_rf), "\n")
cat("重要性大于0的特征数：",
    sum(importance_rf > 0), "\n")
cat("重要性等于0的特征数：",
    sum(importance_rf == 0), "\n")
cat("最小重要性：", min(importance_rf), "\n")
cat("最大重要性：", max(importance_rf), "\n")

cat("重要性分数分位数：\n")
print(quantile(
  importance_rf,
  probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.99, 1),
  na.rm = TRUE
))

# 2. 检查 GRRF 使用的正则化系数
cat("\n【2】coefReg 正则化系数\n")
print(summary(coef_reg))

cat("正则化系数分位数：\n")
print(quantile(
  coef_reg,
  probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.99, 1),
  na.rm = TRUE
))

cat("coefReg 小于0.01的特征数：",
    sum(coef_reg < 0.01), "\n")
cat("coefReg 小于0.1的特征数：",
    sum(coef_reg < 0.1), "\n")

# 3. 检查 GRRF 实际选中的特征
cat("\n【3】GRRF 特征选择结果\n")
cat("feaSet 的内容：\n")
print(grrf$feaSet)

selected_r <- sort(unique(as.integer(grrf$feaSet)))

cat("唯一选中特征数：", length(selected_r), "\n")

# 4. 查看被选中特征的重要性和正则化系数
if (length(selected_r) > 0) {
  print(data.frame(
    feature_index_R = selected_r,
    RF_importance = importance_rf[selected_r],
    coefReg = coef_reg[selected_r]
  ))
}

# 5. 如果存在原始特征名称文件，显示对应名称
features_path <- file.path(data_dir, "features.txt")

if (file.exists(features_path) && length(selected_r) > 0) {
  feature_lines <- readLines(features_path, warn = FALSE)
  feature_names <- sub(
    "^\\s*[0-9]+\\s+",
    "",
    feature_lines
  )

  if (max(selected_r) <= length(feature_names)) {
    cat("\n【4】选中特征对应的原始名称\n")
    print(data.frame(
      feature_index_R = selected_r,
      feature_name = feature_names[selected_r]
    ))
  }
}

cat("\n===== 诊断结束 =====\n")


# 8. 提取 GRRF 选中的特征
selected_r <- sort(unique(as.integer(grrf$feaSet)))

stopifnot(length(selected_r) > 0)
stopifnot(all(selected_r >= 1))
stopifnot(all(selected_r <= 561))

# R 从 1 开始编号，Python 从 0 开始编号
selected_python <- selected_r - 1L

cat("GRRF 选中特征数量：", length(selected_r), "\n")
cat("特征减少比例：",
    round((1 - length(selected_r) / 561) * 100, 2),
    "%\n")

# 9. 保存结果
output_dir <- file.path(root, "results")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(
  data.frame(
    feature_index_R = selected_r,
    feature_index_python = selected_python
  ),
  file.path(output_dir, "grrf_selected_features_initial.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    method = "GRRF_initial_check",
    ntree = ntree,
    mtry = mtry,
    gamma_grrf = gamma_grrf,
    selected_features = length(selected_r)
  ),
  file.path(output_dir, "grrf_run_info_initial.csv"),
  row.names = FALSE
)

cat("GRRF 初步实验完成，结果文件已生成。\n")
