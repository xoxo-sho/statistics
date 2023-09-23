install.packages("ggplot2")
install.packages("caret")

library(dplyr)
library(ggplot2)
library(caret)

df <- read.csv("car-price-prediction/CarPrice_Assignment.csv")
print(df)

####Preprocessing####
df <- df[, setdiff(colnames(df), "car_ID")] # drop car_ID column

X <- df[,1:(ncol(df)) -1]
Y <- df[ncol(df)]

X <- dummyVars(~ ., data = data) # ダミー変数が面倒なので中断
