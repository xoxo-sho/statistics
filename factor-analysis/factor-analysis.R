install.packages("psych")
install.packages("GPArotation")

library(psych)
library(GPArotation)
data(bfi)

dim(bfi)

data <- bfi

data

write.csv(data, file = "date/r_sample_data.csv", row.names = FALSE)
