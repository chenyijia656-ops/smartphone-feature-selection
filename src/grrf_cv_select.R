
# GRRF交叉验证特征选择器
# 输入：当前折训练集的X、y
# 输出：不同gamma下的特征索引及选择时间

library(RRF)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
  stop("需要5个参数：X路径、y路径、特征输出、摘要输出、随机种子")
}

x_path <- args[1]
y_path <- args[2]
features_out <- args[3]
summary_out <- args[4]
seed <- as.integer(args[5])

stopifnot(file.exists(x_path))
stopifnot(file.exists(y_path))

# 1. 读取当前折训练数据
X <- as.matrix(read.csv(
  x_path,
  header = FALSE,
  check.names = FALSE
))

y <- factor