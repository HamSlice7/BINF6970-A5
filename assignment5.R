#BINF6970 Final Assignment by Jacob Hambly
library(glmnet)
library(randomForest)
library(tidyverse)
library(caret)
library(formattable)
###Problem 1

##ai)

#Load in 'forest_fires.csv'
ff_data <- read.csv('forest_fires.csv')

any(is.na(ff_data)) #no missing data


#looking at colinearality - see some NA's
lm_ff <- lm(human_caused ~ ., data = ff_data )

summary(lm_ff)

#remove NA features --> removing wed and precip total. The features precipitation and precipitation total seem to be highly colinear so I remove precipitation total. The days of the week have multicolinearality but removing one of the days removes thes. I chose to remove Wednesday.  Removing these features will help to remove multicollinearality in the data set. 

#re-evaluating the features in the model to ensure to "NA" coefficients. 
lm_ff <- lm(human_caused ~ ., data = ff_data[,-c(69, 32)])
summary(lm_ff)

coef(lm(human_caused ~ ., data = ff_data ))

model.matrix(lm(human_caused ~ ., data = ff_data ))

#removing perfect colinearality (remove wed and total precip)
ff_data <- ff_data[,-c(69, 32)]

#looking at the structure of the data set
str(ff_data)

#checking to see if the features standardized
apply(ff_data[,-c(ncol(ff_data))], 2, mean)
apply(ff_data[,-c(ncol(ff_data))], 2, sd)

#changing certain variables to factors that seem to be binary
ff_data$dayFriday <- as.factor(ff_data$dayFriday)
ff_data$dayMonday <- as.factor(ff_data$dayMonday)
ff_data$daySaturday <- as.factor(ff_data$daySaturday)
ff_data$daySunday <- as.factor(ff_data$daySunday)
ff_data$dayThursday <- as.factor(ff_data$dayThursday)
ff_data$dayTuesday <- as.factor(ff_data$dayTuesday)
ff_data$time_indicator <- as.factor(ff_data$time_indicator)
ff_data$human_caused <- as.factor(ff_data$human_caused)

#scale the non-binary features
non_factor_cols <- sapply(ff_data, function(x) !is.factor(x))

ff_data[, non_factor_cols] <- scale(ff_data[, non_factor_cols])


#split the features from the response
x <- ff_data[, -ncol(ff_data)]
y <- as.factor(ff_data$human_caused)

#looking to see how balance the data set is in terms of non human caused and human caused forest fires 
table(y)


#training and test set - first 10,000 observations are used for the training set and the last 2000 observations are used for the test(validation) set
ff_x_train <- x[1:10000, ]

ff_x_test <- x[10001:12000, ]

ff_y_train <- y[1:10000]

ff_y_test <- y[10001:12000]

###Bootstrapping and Bagging LASSO
set.seed(111)

#number of bootstrapping iterations
B <- 100

#number of rows in ff_x_train
n <- dim(ff_x_train)[1]

#number of columns in ff_x_train
p <- dim(ff_x_train)[2]

#initiating a empty list to contain the sample indices used in each bootstrap
samples <- list()

#initiating an empty list to hold the cvglmnet objects
cvxs <- list()

#creating a progress bar for bootstrapping LASSO 100 times 
pb = txtProgressBar(min = 1, max = B, initial = 0) 

#creating a for loop that creates a lasso logistical regression model from 10,000 randomly selected observations 100 times. The coefficients from the best model are selected, averaged and then ranked which will give insight into feature importance.  
for (i in 1:B) {
  
  samples[[i]] <- sample.int(n, n, replace = TRUE)
  
  cvxs[[i]] <- cv.glmnet(as.matrix(ff_x_train[samples[[i]], ]), ff_y_train[samples[[i]]], family = 'binomial', type.measure = 'auc', alpha = 1,  nfolds = 10)
  
  setTxtProgressBar(pb,i)
}

##creating a matrix of all the 75 beta coefficients associated with the best lambda value for each of the 100 models 
Bcoefs <- matrix(
  unlist(lapply(1:B, function(i){
    coef(cvxs[[i]], s = cvxs[[i]]$lambda.min )[-1,1]
  })), ncol = p, byrow = TRUE
)

#changing the column names of the Bcoefs 
colnames(Bcoefs) <- names(coef(cvxs[[1]])[-1,1])

##rank each of the features based on the number of rows in each column is equal to 0
feature_zero <- apply(Bcoefs, 2, function(x){sum(x == 0)})

feature_zero_sorted <- sort(feature_zero,decreasing = FALSE)

feature_zero_sorted_df <- data.frame(feature_zero_sorted)

#creating a barplot for the features and the number of times they were shrunk towards zero. We can see that nine of the features were never shrunk to zero which indicate that these 9 are the most influencial features
barplot(feature_zero_sorted,ylim=c(0,110), names.arg = "", xlab = "Feature", ylab = "Number of times shrunk to zero by LASSO", main = "Features and the number of times shrunk to zero by LASSO during bootstrapping")


#looking at the 10 features with fewest 0 rows (lowest number of occurs where the feature was push to 0) - indicates importance
feature_zero_sorted[1:10]

#ranking the features based on the mean of the coefficient 
mean_coef <- apply(Bcoefs, 2, mean)

#rank (Top10)
top10_BL <- mean_coef[names(sort(abs(mean_coef),decreasing = TRUE))][1:10]

top10_BL_df <- data.frame(top10_BL)

colnames(top10_BL_df) <- c("Mean Coefficient")

#creating a table of the top 10 features from bagging the boostrapped LASSO models
formattable(top10_BL_df)

###a)ii

#Calculating the ensemble prediction on the test set - bagging the LASSO models
prediction_test_ensemble <- function(cv_object){
  predict(cvxs[[cv_object]], newx = as.matrix(ff_x_test), type = 'response', s = cvxs[[cv_object]]$lambda.min  )[,1]
}

pred <- lapply(1:100,prediction_test_ensemble)

#creating a matrix of predictions 
pred_matrix <- matrix(unlist(pred), ncol = length(pred))

#calculating the mean of each predicted value
pred_matrix_mean <- apply(pred_matrix, 1, mean)

#calculating the AUC for the predicted values from the test set
roc_lasso_bagging <- roc(ff_y_test, pred_matrix_mean) 

#auc for the bolasso model
bolasso_auc <- roc_lasso_bagging$auc[1]

#getting the sensitivities and specificity aloing the ROC curve
roc_lasso_bagging_sn_sp <- cbind(roc_lasso_bagging$sensitivities, roc_lasso_bagging$specificities )

#finding the point on the ROC curve which maximizes sensitivity and specificity 
indx_roc_lasso_bagging_sn_sp <- which.max(apply(roc_lasso_bagging_sn_sp, 1, min))

#finding the optimal sensitivity
lasso_bagging_sensitivity <- roc_lasso_bagging_sn_sp[indx_roc_lasso_bagging_sn_sp,][1]

#finding the optimal specificity 
lasso_bagging_specificity <- roc_lasso_bagging_sn_sp[indx_roc_lasso_bagging_sn_sp,][2]

#finding the optimal threshold based on the optimal sensitivity and the optimal specificity 
optimal_threshold_lasso <- roc_lasso_bagging$thresholds[indx_roc_lasso_bagging_sn_sp]

#plotting ROC for the bagged LASSO predictions with the sensitivities and specifies
plot(roc_lasso_bagging, main = 'ROC Curve for Bolasso')
abline(h = lasso_bagging_sensitivity, v = lasso_bagging_specificity, col = 'blue', lty = 2 )

#generating a confusion matrix for classficiation at the optimal threshold 
pred_matrix_mean_threshold <- ifelse(pred_matrix_mean > optimal_threshold_lasso , 1, 0)

cm_bl <- confusionMatrix(table(pred_matrix_mean_threshold, ff_y_test))
bl_accuracy <- cm_bl$overall[1]

###problem 1 b)

##tuning a random forest model

#deafault mtry
sqrt(p)

#doing a grid search over mtry 6-10 since default is 8.66
#doing a grid seach over nodesize 1-10 at two step iterations as default is 1

rf_tunegrid <- expand.grid(mtry=c(6:10),nodesize=seq(1,10,2))

#implicating a 2 hyperparamter grid search. Method taken from --> https://rpubs.com/phamdinhkhanh/389752

customRF <- list(type = "Classification",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "nodesize"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "nodesize"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               nodesize=param$nodesize)
}

#Predict label
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

#Predict prob
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

#using caret and cv
set.seed(111)
trcontrol <- trainControl(method='repeatedcv', 
                          number=5, 
                          repeats=1, 
                          allowParallel = TRUE)

system.time(rf_gridsearch <- train(as.factor(human_caused) ~ ., 
                                       data = ff_data[1:10000,],
                                       trControl =trcontrol,
                                       method = customRF,
                                       tuneGrid = rf_tunegrid))

#plotting the cv-error of each of the 25 models used
plot(rf_gridsearch)

#mtry = 8, node size = 3 best accuracy

#training a random forest model with optimized parameters 
rf_model_optimized <- randomForest(as.factor(human_caused) ~ ., data = ff_data[1:10000,], mtry = 8, nodesize = 3)

#plotting the number of trees and the looking at the OOB error rates to look for a optimal number of trees. The OOB seem to be leveling out around 200 trees which indicates an optimal number. This will also help reduce overfitting by reducing the complexity of the model.
plot(rf_model_optimized)

rf_model_optimized$err.rate

#re-training the model with an optimal number of trees (200)
rf_model_optimized <- randomForest(as.factor(human_caused) ~ ., data = ff_data[1:10000,], mtry = 8, nodesize = 3, ntree = 200)

#predicting on the test set with the optimized model
rf_optimized_predict <- predict(rf_model_optimized, newdata = ff_data[10001:12000,-ncol(ff_data)], type = 'prob')[,2]

#calculating the AUC for the predictions for the optimized model
roc_optimized_rf <- roc(ff_y_test, rf_optimized_predict) #AIC 0.8112
rf_auc <- roc_optimized_rf$auc[1]

#comparing with the default model. The optimized model seems to have better AUC.
model_rf <- randomForest(as.factor(human_caused) ~ ., data = ff_data[1:10000,])

rf_predict <- predict(model_rf, newdata = ff_data[10001:12000,-ncol(ff_data)], type = 'prob')[,2]

roc(ff_y_test, rf_predict)

##finding optimal threshold of the optimized model
#creating a matrix of sensitivities and specificities for the optimized model
roc_rf_sn_sp <- cbind(roc_optimized_rf$sensitivities, roc_optimized_rf$specificities )

#finding the point on the ROC curve which maximizes sensitivity and specificity 
indx_roc_rf_sn_sp <- which.max(apply(roc_rf_sn_sp, 1, min))

#sensitivity at the optimal threshold
rf_sensitivity <- roc_rf_sn_sp[indx_roc_rf_sn_sp,][1]

#specificity at the optimal threshold
rf_specificity <- roc_rf_sn_sp[indx_roc_rf_sn_sp,][2]

#optimal threshold
optimal_threshold_rf <- roc_optimized_rf$thresholds[indx_roc_rf_sn_sp]

#plotting the ROC curve for the prediction on the test set from the optimized random forest model. The blue lines indicate the sensitivity and specificity.
plot(roc_optimized_rf, main = 'ROC Curve for Random Forest' )
abline(h = rf_sensitivity, v = rf_specificity,  col = 'blue', lty = 2 )

##Finding the accuracy 

#generating a confusion matrix with optimal threshold
pred_rf_threshold <- ifelse(rf_optimized_predict > optimal_threshold_rf , 1, 0)
cm_rf <- confusionMatrix(table(ifelse(rf_optimized_predict > optimal_threshold_rf , 1, 0), ff_y_test))

#extracting the accuracy 
rf_accuracy <- cm_rf$overall[1]


#table of AUC and accuracy from bagging LASSO and the optimized Random Forest model
BL_RF_AUC <- c(bolasso_auc, rf_auc)
BL_RF_accuracy <- c(bl_accuracy, rf_accuracy)

BL_RF_classification_scores <- data.frame(BL_RF_AUC,BL_RF_accuracy )

rownames(BL_RF_classification_scores) <- c("Bagging Lasso", "Random Forest")
colnames(BL_RF_classification_scores) <- c("AUC", "Accuracy")

formattable(BL_RF_classification_scores)

### Problem 1 c)
##feature importance

#plotting the feature importance from the optimized Random Forest model
plot(varImp(rf_gridsearch), cex.names = 0.1)

#getting a table for the top 10 most important features based on how much each predictor contributes to the reduction in accuracy when that predictor is permuted randomly.
feature_importance_rf <- varImp(rf_gridsearch)$importance

as.vector(varImp(rf_gridsearch)$importance[,1])

feature_importance_rf_df <- data.frame(as.vector(varImp(rf_gridsearch)$importance[,1]))

rownames(feature_importance_rf_df) <- rownames(varImp(rf_gridsearch)$importance)
colnames(feature_importance_rf_df) <- c("Importance")

feature_importance_rf_df_sorted <- feature_importance_rf_df %>%
  arrange(desc(Importance))

formattable(feature_importance_rf_df_sorted)

#looking at a simple plot of the most important features
barplot(feature_importance_rf_df_sorted$Gini_Index[1:10], names.arg = rownames(feature_importance_rf_df_sorted)[1:10], las = 2, cex.names = 0.5, horiz = TRUE)

###Problem 2

##i)

#load in the R script and data
load("prob_2_list.RData")

p2_data <- read.csv("problem_2_data.csv")

ls()

str(p2_data)

#seeing if there are any missing values
sum(is.na(p2_data))

#storing each of the linear models generated to 'linear_models'
#initiating an empty list called "linear_models"
linear_models <- list()

#creating a linear model from each of the predictors stored in the 'prob_2_list' iteratively and storing each model to "linear_models"
linear_models <- lapply(1:7, function(i){
  ols <- lm(reformulate(prob_2_list[[i]], response = 'y'), data = p2_data)})

##ii)

#Creating a function to calculate the RMSE. This will be used to calculate the CV-error. o = observed, p = predicted
RMSE <- function(o, p){
  sqrt(mean((o - p)^2))
}

#this will be the number of folds
k <- 10

#Number of observations in each fold
number_per_fold <- 30

#creating 300 fold id's (30 for each fold)
group_assignment <- rep(1:k, each = number_per_fold)


#creating a matrix that to store the average CV-error (RMSE) for each of the 10 repeats for every model
k10_repeat_10 <- matrix(NA, nrow = 10, ncol = 7)
colnames(k10_repeat_10) <- c("Model1", "Model2", "Model3", "Modle4", "Modle5", "Model6", "Model7")

#This block of code executes the cross validation which is made of three for loops in a nested structure. The first loop iterates through each of the repeat's (10 in total). The second for loop iterates through each of the models. The third iterates through each fold in cross validation (10 in total)
for (x in 1:10) {
  
  #take a random sample of 300 from group_assignment to assign each of the observations a random fold
  sample(group_assignment)
  p2_data$fold_id <- sample(group_assignment)
  #initialize a list for mean RMSE
  rmse_list <- list()
  for (i in 1:7) {
    #create a empty vector of 10 elements 
    fold_rmse <- numeric(10)
    for (j in 1:10){
      
      #assign the training and testing folds 
      train <- filter(p2_data, fold_id != j)
      validate <- filter(p2_data, fold_id == j)
      
      #building a linear model on the training set for the current iterating model
      model <- lm(reformulate(prob_2_list[[i]], response = 'y'), data = train)
      #test the model on the testing set
      test_prediction <- as.vector(predict(model, newdata = validate))
      #add the RMSE value to fold_rmse
      fold_rmse[j] <- RMSE(o = validate$y, p = test_prediction )
      
    }
    #Add the 10 RMSE values to rmse_list
    rmse_list[[i]] <- fold_rmse
  }
  
  #find the mean RMSE for each model
  mean_10k <- unlist(lapply(rmse_list, mean))
  #add the mean RMSE to k10_repeat_10
  k10_repeat_10[x,] <- mean_10k
}

#get the means over the 10 repeat's for each model
final_means <- apply(k10_repeat_10, 2, mean)

#plot model complexity and mean CV-error
plot(final_means, main = "Model Complexity and CV-Error (RMSE)", ylab = "CV-Error (RMSE)", xlab = "Model ID")

###

#model AIC of models in 'linear_models' and AIC
aic <- lapply(1:7, function(i){AIC(linear_models[[i]])})
aic <- unlist(aic)
plot(aic, main = "Model Complexity and AIC", ylab = "AIC", xlab = "Model ID")

#model adjusted r-squared of models in 'linear_models'and model complexity
ar2 <- lapply(1:7, function(i){summary(linear_models[[i]])$adj.r.squared})
ar2 <- unlist(ar2)
plot(ar2, main = "Model Complexity and Adjusted R-squared", ylab = "Adjusted R-squared", xlab = "Model ID")
