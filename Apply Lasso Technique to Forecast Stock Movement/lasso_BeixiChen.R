rm(list = ls())
cat("\f")

# Loading packages
library(rJava)
library(xlsx)
library(glmnet)
library(tidyverse)
library(corrplot)

# Readin data
datas = as.matrix(read_xlsx('/Users/bessie.chan/Downloads/Goyal_Welch_Data2016.xlsx',1) )
any(sapply(datas,is.na))
any(sapply(datas,is.nan)) # no NAN or NA

#========================================================================================================================

# setting up
X = datas[,c(6:15)] # 10 variables 
y = datas[,5]  # Excess return as dependent varialbes

startDate <- 196912
estimationEnd = min(which(datas[,"yyyymm"] >= startDate))
TT = nrow(X)
variabletable <- matrix(nrow = 10,ncol = TT - estimationEnd)  # Null table for storing parameters of lasso model
rownames(variabletable) <- colnames(X)
pred_actual <- data.frame(actual = y[(estimationEnd+1):length(y)]) # actual data

# Q1 LASSO models
pred_actual$lasso <- double(nrow(pred_actual)) # Null table for lasso prediction

for (tt in seq(estimationEnd,TT-1)) {
  cv = cv.glmnet(X[1:(tt-1),],y[2:tt],nfolds = 10)  # 10-cross-fold for lasso         
  IndexMinMSE = (as.vector(coef(cv,s='lambda.min'))!=0)[-1]  # choose the lasso result with the minimum MSE
  #All object classes which are returned by model fitting functions (here fitting function is "cv")should provide a coef method or use the default one. (Note that the method is for coef and not coefficients.)
  variabletable[,tt-estimationEnd+1] <- matrix(coef(cv,s='lambda.min')[-1])  # fill the NULL table with parameters
  pred_actual$lasso[tt-estimationEnd+1] <- predict(cv,newx=matrix(X[tt,],nrow =1),s='lambda.min')  # prediction with lasso
}

pred_actual$error <- pred_actual$actual - pred_actual$lasso 
RMSE_lasso = sqrt(mean(pred_actual$error^2))   # errors and RMSE for lasso
plot(pred_actual$actual,type = "l",col = "blue")
lines(pred_actual$lasso,type = "l",col = "red")

# Q2 variable frequency
sort(rowSums(variabletable != 0)/(TT - estimationEnd),decreasing = T) %>%   #rowsum with T/F to calculate the freq  
  barplot()  
#      B.M            net.eq..issue         Inflation           T.bill.rate....year.  long.term.return       E.P 
# 0.310283688          0.212765957          0.195035461          0.177304965          0.170212766          0.152482270
# 
# long.term.yield         D.P              stock.variance       Default.spread 
# 0.060283688          0.017730496          0.007092199          0.000000000

corrplot(cor(X),method ="number") #try to find why the freq of variables aredifferent with Larissa's stepwise, 
# but collineary seems not the reason Cuz not consistent with the different point.
# (we assume if it is collineary, the most show-up variables in two models should relate
# to each other ,but not. Open to answers)

# Q3 prevailing mean models

pred_actual$PM <- double(nrow(pred_actual))

for (i in seq(estimationEnd,TT-1)){
  pred_actual$PM[i-estimationEnd+1] <- mean(y[1:i-1])  # PM(constant) prediction
} 
pred_actual$PM_error <- pred_actual$actual -pred_actual$PM
RMSE_PM = sqrt(mean(pred_actual$PM_error^2))   # errors and RMSE for PM
plot(pred_actual$actual,type = "l",col = "blue")
lines(pred_actual$PM,type = "l",col = "red")

# Q4 Kitchen sink model
pred_actual$KS <- double(nrow(pred_actual))

for (tt in seq(estimationEnd,TT-1)) {
  reg_data <- data.frame(X[1:(tt-1),],y = y[2:tt])   # kitchen sink prediction(AKA, with all variables)
  reg <- lm(y~.,reg_data)
  pred_actual$KS[tt-estimationEnd+1] <- predict(reg,newdata = data.frame(t(X[tt,])),type = "response")
}


pred_actual$KS_error <- pred_actual$actual -pred_actual$KS
RMSE_KS = sqrt(mean(pred_actual$KS_error^2))  # errors and RMSE for kitchen Sink
plot(pred_actual$actual,type = "l",col = "blue")
lines(pred_actual$KS,type = "l",col = "red")

c(lasso = RMSE_lasso, PM = RMSE_PM, KS = RMSE_KS)
# lasso and PM are both similar based on RMSE

# Q5 Economic loss function
pred_actual <- cbind(pred_actual, datas[seq(estimationEnd+1,length(y)),c(16:17)]) %>% 
  mutate(econReturn_lasso = ifelse(lasso>0,stock.return,Risk.free.rate),
         econReturn_PM = ifelse(PM>0,stock.return,Risk.free.rate),
         econReturn_KS = ifelse(KS>0,stock.return,Risk.free.rate))  # create economics return for lasso, PM,and KS 

## lasso econFUN
meanReturns_lasso <- mean(pred_actual$econReturn_lasso)      
sharpeRatios_lasso <- meanReturns_lasso/sd(pred_actual$econReturn_lasso)  

## PM econFUN
meanReturns_PM <- mean(pred_actual$econReturn_PM)
sharpeRatios_PM <- meanReturns_PM/sd(pred_actual$econReturn_PM)  

## KS econFUN
meanReturns_KS <- mean(pred_actual$econReturn_KS)
sharpeRatios_KS <- meanReturns_KS/sd(pred_actual$econReturn_KS)  

## compare
c(lasso = meanReturns_lasso,PM = meanReturns_PM,KS = meanReturns_KS)

#   lasso          PM          KS 
# 0.009516239 0.009241658 0.006462348
# lasso is the best based on econFUN.

c(lasso = sharpeRatios_lasso,PM = sharpeRatios_PM,KS = sharpeRatios_KS)

#   lasso        PM        KS 
# 0.2210931 0.2103606 0.1931200 
# lasso is the best based on sharperatios.

# Q6

#Larissa,it is yours' , haha ^_^!