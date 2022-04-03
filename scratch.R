library(tidyverse)
library(caret)
library(corrplot)
library(rattle)

# read csv, set invalid values to NA
# filter columns with NAs and first 7 columns which are irrelevant to prediction
# add factor to classe
pml.train <- read_csv('./data/pml-training.csv', na = c("", "NA", "#DIV/0!")) %>% 
    select_if(~sum(is.na(.)) == 0) %>%
    select(-(`...1`:num_window)) %>%
    mutate(classe = as.factor(classe))

# check for near-zero variance predictors
nearZeroVar(pml.train, saveMetrics = TRUE)

# split training data into train and validation subsets (80/20)
set.seed(123456) 
inTrain <- createDataPartition(pml.train$classe, p = 0.8, list = FALSE)
train.data <- pml.train[inTrain, ]
train.test <- pml.train[-inTrain, ]
rm(inTrain)
rm(pml.train)

# Build correlation matrix
corr_matrix <- abs(cor(subset(train.data, select = -classe)))

# order by first principal component
corrplot(corr_matrix, method= "square", type = "lower", title = "Correlation Matrix Analysis",
         diag=FALSE, order="FPC", 
         tl.cex=0.55, tl.col="black", tl.srt = 45, 
         cl.pos = 'n', number.cex=0.3, 
         mar=c(0,0,1,0))
rm(corr_matrix)

# look at initial recursive partitioning model
fit.rpart <- train(classe ~ ., data = train.data, method = "rpart")
print(fit.rpart)
dev.off()
fancyRpartPlot(fit.rpart$finalModel, main = NULL, sub = NULL)
confusionMatrix(valid$classe, predict(fit.rpart, train.test))
# only 49.5% accuracy on test set

# preprocess with PCA to reduce collinear predictors
preProc <- preProcess(train.data[, -which(names(train.data) == "classe")], 
                      method="pca", thresh=0.8)
trainPCA <- predict(preProc, train.data[, -which(names(train.data) == "classe")]) %>%
    mutate(classe = train.data$classe)
str(trainPCA)

# use random forest
fit.rf <- train(classe ~ ., data = trainPCA, method = "rf", 
                trControl = trainControl(method = "cv", number = 5))
print(fit.rf)

cf.train <- confusionMatrix(trainPCA$classe, predict(fit.rf, trainPCA))
cf.train
cf.train$overall[1]
#100% accuracy against train.data

#create PCA train.test data set
trainPCA.test <- predict(preProc, train.test[, -which(names(train.test) == "classe")]) %>%
    mutate(classe = train.test$classe)
cf.test <- confusionMatrix(trainPCA.test$classe, predict(fit.rf, trainPCA.test))
cf.test

# test against validation set
pml.validation <- read_csv('./data/pml-testing.csv', na = c("", "NA", "#DIV/0!")) 
predict(fit.rf, predict(preProc, pml.validation))

# Accuracy : 0.9676          
# 95% CI : (0.9616, 0.9729)

#Set preliminary values
l     <- 0.9616
u     <- 0.9729
n     <- nrow(train.test)
alpha <- 0.05;

#Compute sample mean and SD
crit <- qt(alpha/2, df = n-1, lower.tail = FALSE)
MEAN <- (l+u)/2
SD   <- (u-l)*sqrt(n)/(2*crit)
