install.packages("psych")
install.packages("GPArotation")

library(psych)
library(GPArotation)
data(bfi)

data_set = bfi[1:25] # excluding sex,academic background and age

####MAP/BIC####
# https://link.springer.com/article/10.1007/BF02293557
VSS( data_set, n = 8 ) # n is the number of factors than expected
# > MAP 5 factors 

####factor load####
result = fa( date_set, nfactors = 5, fm = "minres", rotate = "promax", use = "complete.obs")
print(result, digits = 3, sort = T)

fa.diagram(result)

print(result$loadings, digits = 2, cutoff = 0.3)
