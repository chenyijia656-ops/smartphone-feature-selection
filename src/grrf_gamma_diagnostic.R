
# GRRF参数敏感性诊断
# 目的：观察不同gamma值对特征选择数量的影响
# 这是诊断实验，不是最终论文复现结果

library(RRF)

# 1. 项目和数据路径
root <- normalizePath(getwd(), winslash = "/")

data_dir <- file.path(
  root, "data", "raw", "UCI HAR Dataset"
)

train_x_path <- file.path(
  data_dir, "train", "X_train.txt"
)

train_y_path <- file.path(
  data_dir, "train", "y_train.txt"
)

features_path <- file.path(
  data_dir, "features.txt"
)

# 2. 检查数据文件
stopifnot(file.exists(train_x_path))
stopifnot(file.exists(train_y_path))
stopifnot(file.exists(features_path))

# 3. 读取官方训练集
X_train <- as.matrix(
  read.table(train_x_path, header = FALSE)
)

y_train <- factor(
  scan(train_y_path, quiet = TRUE)
)

feature_lines <- readLines(
  features_path,
  warn = FALSE
)

feature_names <- sub(
  "^\\s*[0-9]+\\s+",
  "",
  feature_lines
)

stopifnot(nrow(X_train) == 7352)
stopifnot(ncol(X_train) == 561)
stopifnot(length(y_train) == 7352)
stopifnot(length(feature_names) == 561)

cat("训练集维度：", dim(X_train), "\n")

# 4. 固定初步实验参数
ntree <- 100
mtry <- floor(sqrt(ncol(X_train)))

# 比较的gamma值
gamma_values <- c(0, 0.25, 0.5, 0.75, 1)

# 5. 训练普通随机森林，计算特征重要性
set.seed(42)

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

# 6. 归一化重要性
normalized_importance <- (
  importance_rf / max(importance_rf)
)

# 7. 依次运行不同gamma值的GRRF
summary_rows <- list()
selected_rows <- list()

for (g in gamma_values) {

  cat("\n============================\n")
  cat("当前gamma =", g, "\n")

  # 计算引导式正则化系数
  coef_reg <- (
    (1 - g) +
    g * normalized_importance
  )

  # 每个gamma使用相同的随机种子
  set.seed(42)

  grrf <- RRF(
    x = X_train,
    y = y_train,
    ntree = ntree,
    mtry = mtry,
    coefReg = coef_reg,
    flagReg = 1
  )

  # 提取所选特征
  selected_r <- sort(
    unique(as.integer(grrf$feaSet))
  )

  # 检查特征索引是否有效
  stopifnot(length(selected_r) > 0)
  stopifnot(all(selected_r >= 1))
  stopifnot(all(selected_r <= 561))

  n_selected <- length(selected_r)

  reduction_percent <- (
    1 - n_selected / 561
  ) * 100

  cat("选中特征数：", n_selected, "\n")
  cat(
    "特征减少比例：",
    round(reduction_percent, 2),
    "%\n"
  )

  # 保存每组实验的汇总信息
  key <- as.character(g)

  summary_rows[[key]] <- data.frame(
    gamma_grrf = g,
    ntree = ntree,
    mtry = mtry,
    selected_features = n_selected,
    feature_reduction_percent = reduction_percent
  )

  # R索引从1开始，Python索引从0开始
  selected_rows[[key]] <- data.frame(
    gamma_grrf = g,
    feature_index_R = selected_r,
    feature_index_python = selected_r - 1L,
    feature_name = feature_names[selected_r]
  )
}

# 8. 合并所有实验结果
summary_df <- do.call(
  rbind,
  summary_rows
)

selected_df <- do.call(
  rbind,
  selected_rows
)

# 9. 保存结果
output_dir <- file.path(root, "results")

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

summary_path <- file.path(
  output_dir,
  "grrf_gamma_diagnostic_summary.csv"
)

selected_path <- file.path(
  output_dir,
  "grrf_gamma_diagnostic_selected_features.csv"
)

write.csv(
  summary_df,
  summary_path,
  row.names = FALSE
)

write.csv(
  selected_df,
  selected_path,
  row.names = FALSE
)

# 10. 显示最终汇总
cat("\n===== Gamma敏感性实验汇总 =====\n")
print(summary_df)

cat("\n汇总文件：", summary_path, "\n")
cat("特征明细：", selected_path, "\n")

cat("\nGamma敏感性实验完成。\n")
