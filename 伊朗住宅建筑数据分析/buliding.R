install.packages("mlbench")
install.packages("e1071")
install.packages("caret")
install.packages("rpart")
install.packages('seriation')
install.packages('cluster')
install.packages('factoextra')
install.packages('fpc')
install.packages("RWeka")
install.packages('klaR')
install.packages("kernlab")
install.packages("randomForest")
install.packages("ISLR")
install.packages("glmnet")
install.packages('leaps')
install.packages('ggcorrplot')

library(ggplot2)
library(ggcorrplot)
library(corrplot)
library(leaps)
library(ISLR)
library(glmnet)
library(randomForest)
library(readxl)
library(magrittr)
library(ggplot2)
library(lattice)
library(tidyverse)
library(mlbench)
library(e1071)
library(caret)
library(rpart)
library(GGally)
library(seriation)
library(cluster)
library(factoextra)
library(fpc)
library(RWeka)
library(MASS)
library(klaR)
library(kernlab)
library(lattice)
library(Matrix)

buliding<-read_xlsx('C:/Users/pc/Desktop/作业/moding/Residential-Building-Data-Set-new-y1.xlsx')
#buliding<-as_data_frame(scale(buliding))

#####test train
set.seed(2000)
trainlist<-createDataPartition(buliding$y1,p=0.7,list=F)
trainset<-buliding[trainlist,]
testset<-buliding[-trainlist,]

#trainset<-as_data_frame(scale(trainset))
#testset<-as_data_frame(scale(testset))
#####compare different models using raw data

##knn
train_index <- createFolds(trainset$y1, k =10)
knnFit <- trainset %>% train(y1 ~ .,
                             method = "knn",
                             data = .,
                             preProcess = "scale",
                             tuneLength = 5,
                             tuneGrid=data.frame(k = (1:10)),##1 is the best
                             trControl = trainControl(method = "cv", indexOut = train_index))
knnFit
ggplot(knnFit)+geom_point(aes(x= 1,y=231.3908,color='red'),shape = 4,size=5)+geom_point()

##nnet
nnetfit<-trainset %>% train(y1 ~.,
                               method = "nnet",
                               data = .,
                               tuneLength = 5,
                               trControl = trainControl(method = "cv", indexOut = train_index))
nnetfit
ggplot(nnetfit$results,aes(x=nnetfit$results$size,y=nnetfit$results$Rsquared))+geom_point(aes(x= 1,y=0.38262639,color='red'),shape = 4,size=5)+geom_point()##RMSE is same

##ctree
ctreeFit <- trainset %>% train(y1 ~ .,
                                  method = "ctree",
                                  data = .,
                                  tuneLength = 6,
                                  trControl = trainControl(method = "cv", 
                                                           indexOut = train_index))
ctreeFit
ggplot(ctreeFit$results,aes(ctreeFit$results$mincriterion,ctreeFit$results$RMSE))+geom_point(aes(0.01,271.1932,color='red'),shape=4,size=5)+geom_point()+geom_line()

##svm
svmFit <- trainset %>% train(y1 ~.,
                               method = "svmLinear",
                               data = .,
                               tuneLength = 5,
                               trControl = trainControl(method = "cv", indexOut = train_index),tuneGrid = expand.grid(.C = c(1:10)))
svmFit
ggplot(svmFit)+geom_point(aes(10,153.8893,color='red'),shape=4,size=5)

##rf
rffit<-trainset %>% train(y1 ~.,
                            method = "rf",
                            data = .,
                            tuneLength = 5,
                            trControl = trainControl(method = "cv", indexOut = train_index))
rffit
ggplot(rffit)+geom_point(aes(107, 135.4127,color='red'),shape=4,size=5)
###just compare knn is better
resamps <- resamples(list(
  ctree = ctreeFit,
  KNN = knnFit,
  NNET = nnetfit,
  svm = svmFit,
  rf = rffit
))
resamps
bwplot(resamps, layout = c(3, 1))

#########LASSO############################
set.seed(2000)

x=model.matrix(y1~.,buliding)[,-1]
y=unlist(buliding[,108])

cv.out = cv.glmnet(x,y)
bestlam = cv.out$lambda.min
coef = predict(cv.out ,type="coefficients",s=bestlam)
coef

sel = which(coef[,1]!=0)
coef[sel,0]
length(sel)
names(sel[2:36])


#######Reapply strong feature results to the previous model
buliding_strong<-cbind(buliding[names(sel[2:36])],buliding[108])

trainset2<-cbind(trainset[names(sel[2:36])],trainset[108])
testset2<-cbind(testset[names(sel[2:36])],testset[108])

trainset2<-as_data_frame(scale(trainset2))
testset2<-as_data_frame(scale(testset2))
##knn after lasso
train_index2 <- createFolds(trainset2$y1, k =10)
knnFit2 <- trainset2 %>% train(y1 ~ .,
                             method = "knn",
                             data = .,
                             preProcess = "scale",
                             tuneLength = 5,
                             tuneGrid=data.frame(k = (1:10)),
                             trControl = trainControl(method = "cv", indexOut = train_index2))
knnFit2

##rf after lasso
rffit2<-trainset2 %>% train(y1 ~.,
                          method = "rf",
                          data = .,
                          tuneLength = 5,
                          trControl = trainControl(method = "cv", indexOut = train_index2))
rffit2

##svm after lasso
svmFit2 <- trainset2 %>% train(y1 ~.,
                             method = "svmLinear",
                             data = .,
                             tuneLength = 5,
                             trControl = trainControl(method = "cv", indexOut = train_index2),tuneGrid = expand.grid(.C = c(1:10)))
svmFit2

###just compare after lasso
resamps2 <- resamples(list(
  KNN = knnFit,
  KNN2=knnFit2,
  rf = rffit,
  rf2 = rffit2,
  svm = svmFit,
  svm2=svmFit2
))
resamps2
bwplot(resamps2, layout = c(3, 1))

#######mse,mae,R-sequared function set
mse <- function(actual, predicted) {
  mean((actual - predicted)^2)
}
mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}
rsquared <- function(actual, predicted) {
  correlation <- cor(actual, predicted)
  r_squared <- correlation^2
  return(r_squared)
}
####### Test the ability of the LASSO method by training-test strategy
x1=model.matrix(y1~.,trainset)[,-1]
x2=model.matrix(y1~.,testset)[,-1]
y1=unlist(trainset[,108])
cv.out2=cv.glmnet(x1,y1)
bestlam1 = cv.out2$lambda.min

y.pred = predict(cv.out2,x2,type='response',s=bestlam1)
rmse_lasso=sqrt(mean((testset$y1-y.pred)^2))
rmse_lasso ###234.0287 win
mse(testset$y1,y.pred)###54769.42 win
mae(testset$y1,y.pred)###122.634 win
rsquared(testset$y1,y.pred)###0.9696958 win!!!

###knn after lasso ability
y.pred1 = predict(knnFit2,testset2,type='raw')
rmse_lasso_knn=sqrt(mean((testset2$y1-y.pred1)^2))
rmse_lasso_knn ###365.5792
mse(testset2$y1,y.pred1)###133648.2
mae(testset2$y1,y.pred1)### 140.0909
rsquared(testset2$y1,y.pred1)###0.9023555


###what about regsubsets?
result3<-regsubsets(y1~x4+x5+x6+x8+x11+x12+x13+x20+x21+x25+x39+x40+x48+x51+x56+x59+x75+x76+x78+x82+x98+x101,trainset,method = 'backward',nvmax = 23)
result3.5<-regsubsets(y1~x4+x5+x6+x8+x11+x12+x13+x20+x21+x25+x39+x40+x48+x51+x56+x59+x75+x76+x78+x82+x98+x101,trainset,method = 'forward',nvmax = 23)
result3.75<-regsubsets(y1~x4+x5+x6+x8+x11+x12+x13+x20+x21+x25+x39+x40+x48+x51+x56+x59+x75+x76+x78+x82+x98+x101,trainset,nvmax = 23)
result4<-regsubsets(y1~.,trainset)###toooooooo S L O W
summary(result3)$bic ###9 is the best -1065.1721 win!!
summary(result3.5)$bic ###12 is the best -1054.4756
summary(result3.75)$bic

plot(summary(result3)$bic, type="l", ylab="BIC")
lines(summary(result3.5)$bic, col = "red")
lines(summary(result3.75)$bic, col = "blue")
points(9,-1049.4040,col='red')
legend("topright", legend = c("backward", "forward",'bestSubsets'), col = c("black", "red","blue"), lty = 1)

summary(result3.75)$which[9,]

###backward dataset
#buliding_strong_backward<-cbind(building[c('x4','x11','x12','x21','x39','x48','x59','x76','x78')],buliding[108])

############################################################backward ability
trainset3<-cbind(trainset2[c('x4','x11','x12','x21','x39','x48','x59','x76','x78')],trainset2['y1'])
testset3<-cbind(testset2[c('x4','x11','x12','x21','x39','x48','x59','x76','x78')],testset2['y1'])

trainset3<-cbind(trainset2[c('x8','x9','x20','x22','x37','x48','x54','x96','x101')],trainset2['y1'])
testset3<-cbind(testset2[c('x8','x9','x20','x22','x37','x48','x54','x96','x101')],testset2['y1'])


##knn after backward
train_index3 <- createFolds(trainset3$y1, k =10)
knnFit3 <- trainset3 %>% train(y1 ~ .,
                               method = "knn",
                               data = .,
                               preProcess = "scale",
                               tuneLength = 5,
                               tuneGrid=data.frame(k = (1:10)),
                               trControl = trainControl(method = "cv", indexOut = train_index3))
knnFit3

##rf after backward
rffit3<-trainset3 %>% train(y1 ~.,
                            method = "rf",
                            data = .,
                            tuneLength = 5,
                            trControl = trainControl(method = "cv", indexOut = train_index3))
rffit3

##svm after backward
svmFit3 <- trainset3 %>% train(y1 ~.,
                               method = "svmLinear",
                               data = .,
                               tuneLength = 5,
                               trControl = trainControl(method = "cv", indexOut = train_index3),tuneGrid = expand.grid(.C = c(1:10)))
svmFit3

###just compare after backward
resamps3 <- resamples(list(
  KNN = knnFit,
  KNN2=knnFit2,
  KNN3=knnFit3,
  rf = rffit,
  rf2 = rffit2,
  rf3 = rffit3,
  svm = svmFit,
  svm2=svmFit2,
  svm3=svmFit3
))
bwplot(resamps3, layout = c(3, 1))

###what about lm?
result5<-lm(y1~.,trainset3)
anova(result5)

y.pred2<-predict(result5,testset3)
rmse_lm=sqrt(mean((testset3$y1-y.pred2)^2))
rmse_lm 
mse(testset3$y1,y.pred2)
mae(testset3$y1,y.pred2)
rsquared(testset3$y1,y.pred2)


###################y2
#lasso
buliding2<-read_xlsx('C:/Users/sili/Desktop/buliding/buliding 1122/Residential-Building-Data-Set-new-y2.xlsx')

set.seed(2000)

x2=model.matrix(y2~.,buliding2)[,-1]
y2=unlist(buliding2[,108])

cv.out2 = cv.glmnet(x2,y2)
bestlam2 = cv.out2$lambda.min
coef2 = predict(cv.out2 ,type="coefficients",s=bestlam2)
coef2

sel2 = which(coef2[,1]!=0)
coef2[sel2,0]
names(sel2[2:40])
#"x2"  "x4"  "x5"  "x7"  "x8"  "x9"  "x10" "x11" "x16" "x20" "x22" "x23" "x25" "x26" "x31" "x35" "x37" "x40" "x41" "x48" "x49" 
#"x54"  "x56"  "x59"  "x70"  "x73"  "x77"  "x78"  "x80"  "x82"  "x89"  "x95"  "x96"  "x97"  "x98"  "x100" "x101" "x102" "x105"
length(sel2)###40

#bestsubsets
result_y2<-regsubsets(y2~x2+x4+x5+x7+x8+x9+x10+x11+x16+x20+x22+x23+x25+x26+x31+x35+x37+x40+x41+x48+x49+x54+x56+x59+x70+x73+x77+x78+x80+x82+x89+x95+x96+x97+x98+x100+x101+x102+x105,buliding2,nvmax = 40)
min(summary(result_y2)$bic)#12 is the smallest
summary(result3.75)$which[12,]
plot(summary(result_y2)$bic, type="l", ylab="BIC")

########################cov
###cov between V-19 in leg1,3,4 and V-9,V-10
buliding3<-read_xlsx('C:/Users/pc/Desktop/作业/moding/Residential-Building-Data-Set-new.xlsx')

buliding19=buliding3[,c('x21','x59','x78','y1','y2')]
buliding19_scal=scale(buliding19)
ggcorrplot(cor(buliding19_scal),colors = c("CornflowerBlue","white","Salmon"),lab = T,lab_size = 3,tl.cex = 8)
ggcorrplot(cor(buliding19_scal), method = 'circle',type = 'lower',outline.color = 'white',lab_size = 6)
###cov between V-48 in leg2 and V-9,V-10
buliding27=buliding3[,c('x48','y1','y2')]
buliding27_scal=scale(buliding27)
cor(buliding27_scal)
ggcorrplot(cor(buliding27_scal), method = 'circle',type = 'lower',outline.color = 'white',lab_size = 6)
###cov between V-14 in leg1 and V-9
buliding51=buliding3[,c('x16','y2')]
buliding51_scal=scale(buliding51)
ggcorrplot(cor(buliding51_scal), method = 'circle',type = 'lower',outline.color = 'white',lab_size = 6)