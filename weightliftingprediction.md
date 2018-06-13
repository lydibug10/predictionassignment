---
title: "Weight Lifting Prediction Assignment"
author: "Lydia Cromwell"
date: "June 9, 2018"
output: 
  html_document:
    keep_md: yes
    self_contained: yes
---

# Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# The Data

The data is source from http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har


```r
trainingData <- read.csv("pml-training.csv")
testingData <- read.csv("pml-testing.csv")
```

We cleanse the data by removing N/As and other unnecessary columns otherwise the model algorithm will fail. This will also shrink the dataset so the model runs quicker.

```r
# Remove NA values
trainingData <- trainingData[,colSums(is.na(trainingData)) == 0]
testingData <- testingData[,colSums(is.na(testingData)) == 0]
classe <- trainingData$classe
# Remove timestamp and extra data
trainingData <- trainingData[,8:59]
trainingData <- trainingData[, sapply(trainingData, is.numeric)]
testingData <- testingData[,8:59]
testingData <- testingData[, sapply(testingData, is.numeric)]
trainingData$classe <- classe
```

We then partition the training data 60/40

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(20180612)
trainPartition <- createDataPartition(trainingData$classe, p=0.6, list=FALSE)
trainingDataPartition <- trainingData[trainPartition,]
testingDataPartition <- trainingData[-trainPartition,]
```

# Data Modeling
We use the GBM algorithm for modeling the data and building a decision tree.

```r
# Use cv resampling mode and GBM algorithm to train the model
control <- trainControl(method="cv", number=5, repeats=1)
gbmmodel <- train(classe ~ ., data=trainingDataPartition, method="rf", trControl=control, verbose=FALSE)
gbmmodel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, verbose = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 15
## 
##         OOB estimate of  error rate: 2.01%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3283   20   10   23   12 0.019414576
## B   22 2227   26    3    1 0.022817025
## C    6   30 1994   22    2 0.029211295
## D    9    0   29 1891    1 0.020207254
## E    1   11    4    5 2144 0.009699769
```

```r
prediction <- predict(gbmmodel, newdata=testingDataPartition)
confusionMatrix(testingDataPartition$classe, prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2156   21   10   30   15
##          B   18 1476   20    0    4
##          C    5   11 1338   14    0
##          D    5    0   26 1251    4
##          E    0    4    6    5 1427
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9748         
##                  95% CI : (0.971, 0.9781)
##     No Information Rate : 0.2784         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9681         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9872   0.9762   0.9557   0.9623   0.9841
## Specificity            0.9866   0.9934   0.9953   0.9947   0.9977
## Pos Pred Value         0.9659   0.9723   0.9781   0.9728   0.9896
## Neg Pred Value         0.9950   0.9943   0.9904   0.9925   0.9964
## Prevalence             0.2784   0.1927   0.1784   0.1657   0.1848
## Detection Rate         0.2748   0.1881   0.1705   0.1594   0.1819
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9869   0.9848   0.9755   0.9785   0.9909
```

This GBM model gives a 97.48% Accuracy which is pretty good with an error of ~3%. Let's now apply that to the test data.

# Apply the model to the TestData
We apply our model to the test data and use these answers for the quiz

```r
predict(gbmmodel, newdata=testingData)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
