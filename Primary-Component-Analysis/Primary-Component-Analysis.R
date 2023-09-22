library(dplyr)

data(bfi)

data_set = bfi[1:25] # excluding sex,academic background and age

data_set <- na.omit(data_set) # excluding rows with missing value

# 主成分分析を実行
pca <- princomp(data_set)
summary(pca)

# 主成分得点

# 第1主成分とデータセットを計算
z <- pca$rotation
print(z)

# 結果を四捨五入してデータフレームに変換
# z <- round(z, 2) %>% t() %>% as.data.frame()
# colnomes(z) <- "z"
# z %>% head()
