
library(vctrs)
library(rsample)      
library(gbm)          
library(xgboost)      
library(caret)       
library(h2o)          
library(pdp)          
library(ggplot2)      
library(lime)         
library(vip)
library(dplyr)
library(pROC)
library(gcookbook)
library(cvAUC)
library(missForest)
library(mlbench)
library(caret)
library(ggpubr)
library(ROCR)
library(mlr)
library("Ckmeans.1d.dp") 
library(multiROC)
library(VIM)
########################################################
data1<-read.csv("Test_346_AMSY_4category_add.csv")
md.pattern(data1) 

set.seed(0304)

data11<-data1
data11$weight<-as.numeric(data11$weight)
data11$Bend_class<-as.factor(data11$Bend_class)
data11$LH<-as.factor(data11$LH)
data11$Res<-as.factor(data11$Res)
data11$M_k<-as.numeric(data11$M_k)
data11$Ntype<-as.factor(data11$Ntype)
data11$Linf<-as.numeric(data11$Linf)
data11$k <-as.numeric( data11$k)
data11$Winf <- as.numeric(data11$Winf)
data11$tmax <-as.numeric(data11$tmax)
data11$tm <-as.numeric(data11$tm)
data11$M <-as.numeric(data11$M)
data11$Lm <-as.numeric(data11$Lm)
data11$R_var <-as.numeric(data11$R_var)
data11$rho <-as.numeric( data11$rho)
data11$h <-as.numeric(data11$h)
data11$G <-as.numeric( data11$G)
data11$Nend_ratio <- as.numeric(data11$Nend_ratio)
data11$ Nnyrs <- as.integer(data11$Nnyrs)
data11$ave_Nratio <- as.numeric(data11$ave_Nratio)
data11$cv_Nratio <- as.numeric(data11$cv_Nratio)
data11$num_bigN <- as.integer(data11$num_bigN)
data11$max_min_Nratio<-as.numeric(data11$max_min_Nratio)
data11$Nave_lst5_ratio<-as.numeric(data11$Nave_lst5_ratio)

data11$DR_true<-as.factor(data11$DR_true)
data11$Habitat <- as.factor(data11$Habitat)
data11$Area <-as.factor(data11$Area)
data11$Ass_method <- as.factor(data11$Ass_method)
data11$TACyn <- as.factor(data11$TACyn)

data11$FMI <- as.numeric(data11$FMI)

set.seed(0304)
setDT(data11)
fish_split <- initial_split(data11, prop=.80) 
fish_test <-testing(fish_split)
fish_train <-training(fish_split)
labels <- fish_train$DR_true
ts_label <- fish_test$DR_true
all_label <-data11$DR_true
weights <-fish_train$weight
ts_weight <-fish_test$weight
all_weight <-data11$weight

options(na.action='na.pass')
new_tr <- model.matrix(~.+0,data = fish_train[,-c("DR_true","weight"),with=F])#one hot encoding
new_ts <- model.matrix(~.+0,data = fish_test[,-c("DR_true","weight"),with=F])
all_data <-model.matrix(~.+0,data=data11[,-c("DR_true","weight"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
all_label <-as.numeric(all_label)-1


#preparing matrix
dtrain <- xgb.DMatrix(data = new_tr,label = labels,missing=NaN, weight=weights) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label,missing=NaN, weight=ts_weight)
dll <- xgb.DMatrix(data = all_data,label=all_label,missing=NaN, weight=all_weight)

########################################################
#try tuning with cross validation
start.time <- Sys.time()
numberOfClasses <- length(unique(labels))
# Create empty lists
lowest_error_list = list()
parameters_list = list()
auc_list =list()

cv.nround = 1000 #1000
cv.nfold = 5
# Create 10,000 rows with random hyperparameters
set.seed(0304)
for (iter in 1:5000){
  param <- list(booster = "gbtree",
                objective = "multi:softprob",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .5, 0.8),
                colsample_bytree = runif(1, .6, 1),
                eval_metric="mlogloss", 
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  #for (row in 1:10){ 
  set.seed(0304)
  mdcv <- xgb.cv(data= dtrain,
                    booster = "gbtree",
                   objective = "multi:softprob",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= cv.nround,
                    nfold=cv.nfold,
                 eval_metric="mlogloss",
                 num_class= numberOfClasses,
                 nthread=6,
                 early_stopping_rounds = 100,
                 verbose =F
  )
  lowest_error <- as.data.frame(min(mdcv$evaluation_log$test_mlogloss_mean))
  lowest_error_list[[row]] <- lowest_error
  print(row)
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_error_df, parameters_df)

# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

write.csv(randomsearch, "randomsearch_alldata0304notemp.csv")
#up to here is tuning---
########################################################
# apply the best tuning pars 
# note that the best tuning pars must be applied manually
set.seed(0304)
xgb_params <- list(objective = "multi:softprob",
              booster = "gbtree",
              eval_metric = "mlogloss",
              num_class = 4,
              max_depth = 3,
              eta = 0.199,
              #gamma = 0.01, 
              subsample = 0.6346,
              colsample_bytree = 0.7826, 
              min_child_weight = 0
              #max_delta_step = 1
)
cv.nround = 1000
cv.nfold = 5
mdcv <- xgb.cv(data=dtrain, params = xgb_params , nthread=6, 
               nfold=cv.nfold, nrounds=cv.nround,
               verbose = T)

min_logloss = min(mdcv$evaluation_log$test_mlogloss_mean)
min_logloss_index = which.min(mdcv$evaluation_log$test_mlogloss_mean)
min_logloss
min_logloss_index

set.seed(0304)

numberOfClasses <- length(unique(labels))
bst_model <- xgb.train(data = dtrain,
                       nrounds= min_logloss_index,
                       params = xgb_params
)

########################################################
# test1: Apply the best model and predict for test data---
test_pred <- predict(bst_model, newdata = dtest)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = ts_label + 1,
         max_prob = max.col(., "last"))

write.csv(test_prediction,"test_prediction.csv")
# confusion matrix of test set
confusionMatrix(data=factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")

#calculate multiclass auc
roc.multi<-multiclass.roc(predictor=test_prediction$max_prob,  response=test_prediction$label,levels=c(1,2,3,4))
auc(roc.multi)

rs <- roc.multi[['rocs']]
plot.roc(rs[[1]])
plot.roc(rs[[1]],legacy.axes=TRUE,xaxs="i",yaxs="i",xlim=c(1.00,0.00),ylim=c(0.00,1.00),xlab="False Positive Rate",ylab="True Positive Rate")
lines.roc(rs[[2]],col=2)
lines.roc(rs[[3]],col=3)
lines.roc(rs[[4]],col=4)
lines.roc(rs[[5]],col=5)
lines.roc(rs[[6]],col=6)
#sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))
legend(x="bottomright",
       legend=c("1vs2","1vs3","1vs4","2vs3","2vs4","3vs4"),
       lty=c(1,1,1,1,1,1),
       col=c(1,2,3,4,5,6))

x<-c(auc(rs[[1]]),auc(rs[[2]]),auc(rs[[3]]),auc(rs[[4]]),auc(rs[[5]]),auc(rs[[6]]))
mean(x)

########################################################
# test2: Apply the best model and predict for the whole data---
test_pred <- predict(bst_model, newdata = dll)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = all_label + 1,
         max_prob = max.col(., "last"))

# confusion matrix of test set
confusionMatrix(data=factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")

#calculate multiclass auc
roc.multi<-multiclass.roc(predictor=test_prediction$max_prob,  response=test_prediction$label,levels=c(1,2,3,4))
auc(roc.multi)

rs <- roc.multi[['rocs']]
plot.roc(rs[[1]])
plot.roc(rs[[1]],legacy.axes=TRUE,xaxs="i",yaxs="i",xlim=c(1.00,0.00),ylim=c(0.00,1.00),xlab="False Positive Rate",ylab="True Positive Rate")

#sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))
legend(x="bottomright",
       legend=c("1vs2","1vs3","1vs4","2vs3","2vs4","3vs4"),
       lty=c(1,1,1,1,1,1),
       col=c(1,2,3,4,5,6))

x<-c(auc(rs[[1]]),auc(rs[[2]]),auc(rs[[3]]),auc(rs[[4]]),auc(rs[[5]]),auc(rs[[6]]))
mean(x)

########################################################
#do importance_plot plottings---
importance_matrix = xgb.importance( model = bst_model)
head(importance_matrix)

mat<- xgb.importance( model = bst_model)
xgb.plot.importance(importance_matrix=mat[1:33])

f1<-vip(bst_model,num_features = 35) #this is also ok

f<-vi(bst_model)
f<-as.data.frame(f)
length(f$Variable)
ggsave(f1,filename="AMSY_VI_notemp.png",width=10,height=6)
write.csv(f,"AMSY_VI_0304_notemp.csv")
