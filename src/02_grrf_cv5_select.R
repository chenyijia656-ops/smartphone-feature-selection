suppressPackageStartupMessages(library(RRF))
root <- normalizePath(getwd(), winslash = "/")
data_dir <- file.path(root, "data", "raw", "UCI HAR Dataset", "train")
out_dir <- file.path(root, "results")
split_file <- file.path(out_dir, "grrf_cv5_splits.csv")
stopifnot(file.exists(split_file))
X <- as.matrix(read.table(file.path(data_dir, "X_train.txt"), header = FALSE))
y <- factor(scan(file.path(data_dir, "y_train.txt"), quiet = TRUE))
subjects <- scan(file.path(data_dir, "subject_train.txt"), quiet = TRUE)
splits <- read.csv(split_file)
stopifnot(nrow(X) == 7352, ncol(X) == 561, length(y) == 7352)
stopifnot(nrow(splits) == 7352, identical(splits$row_index_python, 0:7351))
stopifnot(all(splits$subject == subjects), setequal(splits$validation_fold, 1:5))

ntree <- 100L  # 诊断参数，不是已确认的论文最优参数
mtry <- floor(sqrt(ncol(X)))
gamma_values <- c(0.25, 1.0)  # 对照早期结果；论文使用 gamma=1
selected_rows <- list()
summary_rows <- list()
for (fold in 1:5) {
  train_rows <- which(splits$validation_fold != fold)
  val_rows <- which(splits$validation_fold == fold)
  stopifnot(length(intersect(unique(subjects[train_rows]), unique(subjects[val_rows]))) == 0)
  Xtr <- X[train_rows, , drop = FALSE]
  ytr <- y[train_rows]
  set.seed(1000 + fold)
  rf_start <- proc.time()[3]
  rf <- RRF(x = Xtr, y = ytr, ntree = ntree, mtry = mtry,
            flagReg = 0, importance = TRUE)
  rf_seconds <- as.numeric(proc.time()[3] - rf_start)
  importance_rf <- rf$importance[, "MeanDecreaseGini"]
  stopifnot(length(importance_rf) == 561, all(is.finite(importance_rf)), max(importance_rf) > 0)
  normalized_importance <- importance_rf / max(importance_rf)

  for (g in gamma_values) {
    coef_reg <- (1 - g) + g * normalized_importance
    set.seed(2000 + fold)  # 同一折不同gamma的随机种子相同
    begin <- proc.time()[3]
    model <- RRF(x = Xtr, y = ytr, ntree = ntree, mtry = mtry,
                 flagReg = 1, coefReg = coef_reg)
    grrf_seconds <- as.numeric(proc.time()[3] - begin)
    features_R <- sort(unique(as.integer(model$feaSet)))
    stopifnot(length(features_R) > 0, all(features_R >= 1), all(features_R <= 561))
    key <- paste(fold, g, sep = "_")
    selected_rows[[key]] <- data.frame(
      fold = fold, gamma_grrf = g, feature_index_python = features_R - 1L)
    summary_rows[[key]] <- data.frame(
      fold = fold, gamma_grrf = g, selected_features = length(features_R),
      rf_importance_seconds = rf_seconds, grrf_seconds = grrf_seconds,
      selection_total_seconds = rf_seconds + grrf_seconds)
    cat(sprintf("折 %d, gamma=%.2f: %d 个特征, 选择耗时 %.2fs\n",
                fold, g, length(features_R), rf_seconds + grrf_seconds))
  }
}
write.csv(do.call(rbind, selected_rows),
          file.path(out_dir, "grrf_cv5_selected_features.csv"), row.names = FALSE)
write.csv(do.call(rbind, summary_rows),
          file.path(out_dir, "grrf_cv5_selection_times.csv"), row.names = FALSE)
cat("每折特征选择完成。\n")
