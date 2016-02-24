---
title: "Machine Learning Assignment"
author: "Zuraida"
date: "Monday, February 22, 2016"
output: html_document
---

PRACTICAL MACHINE LEARNING PROJECT
======================================
The goal of THE project is to predict the manner in which the participant did the exercise. This is the "classe" variable in the training set. The prediction model is described in details in the following section and it is used to predict 20 different test cases.

Getting The Data 
================
The data from the url given is dowloaded as follows:

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#setwd("D:/R assignment/Practical machine learning/Assignment2")

#create the folder if it is not exist in the directory
#if(!file.exists("data")){
 # dir.create("data")
#}

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
getwd()
```

```
## [1] "C:/Users/User/Desktop/MachineLearning"
```


Pre-processing the data
=======================
The data need to be pre-processed.This is done by deleting unnessary column such as column representing ID, columns that contain too much NA as well as column that has near zero variance.


```r
col <- length(training) #no of column
row <-nrow(training)  #no of row

#(a)Remove the first column of the training data - the ID variable 
mytraining <- training[c(-1)]
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)

myDataNZV <- nearZeroVar(mytraining, saveMetrics=TRUE)
myNZVvars <- names(mytraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                      "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                      "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                      "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                      "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                      "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                      "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                      "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                      "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                      "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                      "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                      "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                      "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                      "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                      "stddev_yaw_forearm", "var_yaw_forearm")
mytraining <- mytraining[!myNZVvars]

#(b)Remove the coloum with more than 60% NA.
naVector <- vector()

for(i in 1:length(mytraining)){
  if(sum(is.na(mytraining[ ,i]))/row > 0.6)
  {
    naVector <- c(naVector, i)
   }
}

mytrainingV2 <- mytraining[ ,-naVector]
```

Data partitioning & Cross Validation
=====================================
The training data will be divided into training set and testing set. Cross validation will be used.

```r
library(caret)
inTrain <- createDataPartition(y=mytrainingV2$classe, p=0.6, list=FALSE)
training_dataset <- mytrainingV2[inTrain, ]
CV <- mytrainingV2[-inTrain, ]

inTrain2 <- createDataPartition(y=CV$classe, p=0.5, list=FALSE)
CVtraining_dataset <- CV[inTrain2, ]
CVtesting_dataset <- CV[-inTrain2, ]

dim(training_dataset)
```

```
## [1] 11776    58
```

```r
dim(CVtraining_dataset)
```

```
## [1] 3923   58
```

```r
dim(CVtesting_dataset)
```

```
## [1] 3923   58
```

Apply Machine Learning & Cross Validation: Random Forest and the prediction 
============================================================================
The machine learning algorithm which is Randon Forest is applied for prediction of how well the participant do the exercise. The reason why Random Forest is choosed is because it is one of the most used/accurate algorithms along with boosting.


```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
rfFit <- randomForest(classe ~., data=training_dataset, type = "class") 
rfPredict <- predict(rfFit, CVtraining_dataset)
confusionMatrix (rfPredict, CVtraining_dataset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1113    0    0    0    0
##          B    3  759    1    0    0
##          C    0    0  682    5    0
##          D    0    0    1  638    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9972         
##                  95% CI : (0.995, 0.9986)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9965         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   1.0000   0.9971   0.9922   0.9986
## Specificity            1.0000   0.9987   0.9985   0.9994   1.0000
## Pos Pred Value         1.0000   0.9948   0.9927   0.9969   1.0000
## Neg Pred Value         0.9989   1.0000   0.9994   0.9985   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1935   0.1738   0.1626   0.1835
## Detection Prevalence   0.2837   0.1945   0.1751   0.1631   0.1835
## Balanced Accuracy      0.9987   0.9994   0.9978   0.9958   0.9993
```

```r
rfPredict1 <- predict(rfFit, CVtesting_dataset)
confusionMatrix (rfPredict1, CVtesting_dataset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    0
##          B    1  759    0    0    0
##          C    0    0  683    0    0
##          D    0    0    1  643    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9982, 0.9999)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   1.0000   0.9985   1.0000   1.0000
## Specificity            1.0000   0.9997   1.0000   0.9997   1.0000
## Pos Pred Value         1.0000   0.9987   1.0000   0.9984   1.0000
## Neg Pred Value         0.9996   1.0000   0.9997   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1935   0.1741   0.1639   0.1838
## Detection Prevalence   0.2842   0.1937   0.1741   0.1642   0.1838
## Balanced Accuracy      0.9996   0.9998   0.9993   0.9998   1.0000
```

The model is used for prediction using the testing dataset. The testing data set need to pre-process as follows:


```r
mytesting <- testing[c(-1)]
myDataNZV <- nearZeroVar(mytesting, saveMetrics=TRUE)
myNZVvars <- names(mytesting) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                      "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                      "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                      "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                      "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                      "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                      "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                      "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                      "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                      "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                      "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                      "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                      "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                      "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                      "stddev_yaw_forearm", "var_yaw_forearm")
mytesting <- mytesting[!myNZVvars]
rowt <- nrow(mytesting) # number of row for testing dataset

#(b)Remove the coloum with more than 60% NA.
naVector <- vector()

for(i in 1:length(mytesting)){
  if(sum(is.na(mytesting[ ,i]))/rowt > 0.6)
  {
    naVector <- c(naVector, i)
  }
}

mytestingV2 <- mytesting[ ,-naVector]
summary(mytestingV2)
```

```
##     user_name raw_timestamp_part_1 raw_timestamp_part_2
##  adelmo  :1   Min.   :1.322e+09    Min.   : 36553      
##  carlitos:3   1st Qu.:1.323e+09    1st Qu.:268655      
##  charles :1   Median :1.323e+09    Median :530706      
##  eurico  :4   Mean   :1.323e+09    Mean   :512167      
##  jeremy  :8   3rd Qu.:1.323e+09    3rd Qu.:787738      
##  pedro   :3   Max.   :1.323e+09    Max.   :920315      
##                                                        
##           cvtd_timestamp   num_window      roll_belt       
##  30/11/2011 17:11:4      Min.   : 48.0   Min.   : -5.9200  
##  05/12/2011 11:24:3      1st Qu.:250.0   1st Qu.:  0.9075  
##  30/11/2011 17:12:3      Median :384.5   Median :  1.1100  
##  05/12/2011 14:23:2      Mean   :379.6   Mean   : 31.3055  
##  28/11/2011 14:14:2      3rd Qu.:467.0   3rd Qu.: 32.5050  
##  02/12/2011 13:33:1      Max.   :859.0   Max.   :129.0000  
##  (Other)         :5                                        
##    pitch_belt         yaw_belt      total_accel_belt  gyros_belt_x   
##  Min.   :-41.600   Min.   :-93.70   Min.   : 2.00    Min.   :-0.500  
##  1st Qu.:  3.013   1st Qu.:-88.62   1st Qu.: 3.00    1st Qu.:-0.070  
##  Median :  4.655   Median :-87.85   Median : 4.00    Median : 0.020  
##  Mean   :  5.824   Mean   :-59.30   Mean   : 7.55    Mean   :-0.045  
##  3rd Qu.:  6.135   3rd Qu.:-63.50   3rd Qu.: 8.00    3rd Qu.: 0.070  
##  Max.   : 27.800   Max.   :162.00   Max.   :21.00    Max.   : 0.240  
##                                                                      
##   gyros_belt_y     gyros_belt_z      accel_belt_x     accel_belt_y   
##  Min.   :-0.050   Min.   :-0.4800   Min.   :-48.00   Min.   :-16.00  
##  1st Qu.:-0.005   1st Qu.:-0.1375   1st Qu.:-19.00   1st Qu.:  2.00  
##  Median : 0.000   Median :-0.0250   Median :-13.00   Median :  4.50  
##  Mean   : 0.010   Mean   :-0.1005   Mean   :-13.50   Mean   : 18.35  
##  3rd Qu.: 0.020   3rd Qu.: 0.0000   3rd Qu.: -8.75   3rd Qu.: 25.50  
##  Max.   : 0.110   Max.   : 0.0500   Max.   : 46.00   Max.   : 72.00  
##                                                                      
##   accel_belt_z     magnet_belt_x    magnet_belt_y   magnet_belt_z   
##  Min.   :-187.00   Min.   :-13.00   Min.   :566.0   Min.   :-426.0  
##  1st Qu.: -24.00   1st Qu.:  5.50   1st Qu.:578.5   1st Qu.:-398.5  
##  Median :  27.00   Median : 33.50   Median :600.5   Median :-313.5  
##  Mean   : -17.60   Mean   : 35.15   Mean   :601.5   Mean   :-346.9  
##  3rd Qu.:  38.25   3rd Qu.: 46.25   3rd Qu.:631.2   3rd Qu.:-305.0  
##  Max.   :  49.00   Max.   :169.00   Max.   :638.0   Max.   :-291.0  
##                                                                     
##     roll_arm         pitch_arm          yaw_arm        total_accel_arm
##  Min.   :-137.00   Min.   :-63.800   Min.   :-167.00   Min.   : 3.00  
##  1st Qu.:   0.00   1st Qu.: -9.188   1st Qu.: -60.15   1st Qu.:20.25  
##  Median :   0.00   Median :  0.000   Median :   0.00   Median :29.50  
##  Mean   :  16.42   Mean   : -3.950   Mean   :  -2.80   Mean   :26.40  
##  3rd Qu.:  71.53   3rd Qu.:  3.465   3rd Qu.:  25.50   3rd Qu.:33.25  
##  Max.   : 152.00   Max.   : 55.000   Max.   : 178.00   Max.   :44.00  
##                                                                       
##   gyros_arm_x      gyros_arm_y       gyros_arm_z       accel_arm_x    
##  Min.   :-3.710   Min.   :-2.0900   Min.   :-0.6900   Min.   :-341.0  
##  1st Qu.:-0.645   1st Qu.:-0.6350   1st Qu.:-0.1800   1st Qu.:-277.0  
##  Median : 0.020   Median :-0.0400   Median :-0.0250   Median :-194.5  
##  Mean   : 0.077   Mean   :-0.1595   Mean   : 0.1205   Mean   :-134.6  
##  3rd Qu.: 1.248   3rd Qu.: 0.2175   3rd Qu.: 0.5650   3rd Qu.:   5.5  
##  Max.   : 3.660   Max.   : 1.8500   Max.   : 1.1300   Max.   : 106.0  
##                                                                       
##   accel_arm_y      accel_arm_z       magnet_arm_x      magnet_arm_y   
##  Min.   :-65.00   Min.   :-404.00   Min.   :-428.00   Min.   :-307.0  
##  1st Qu.: 52.25   1st Qu.:-128.50   1st Qu.:-373.75   1st Qu.: 205.2  
##  Median :112.00   Median : -83.50   Median :-265.00   Median : 291.0  
##  Mean   :103.10   Mean   : -87.85   Mean   : -38.95   Mean   : 239.4  
##  3rd Qu.:168.25   3rd Qu.: -27.25   3rd Qu.: 250.50   3rd Qu.: 358.8  
##  Max.   :245.00   Max.   :  93.00   Max.   : 750.00   Max.   : 474.0  
##                                                                       
##   magnet_arm_z    roll_dumbbell      pitch_dumbbell    yaw_dumbbell      
##  Min.   :-499.0   Min.   :-111.118   Min.   :-54.97   Min.   :-103.3200  
##  1st Qu.: 403.0   1st Qu.:   7.494   1st Qu.:-51.89   1st Qu.: -75.2809  
##  Median : 476.5   Median :  50.403   Median :-40.81   Median :  -8.2863  
##  Mean   : 369.8   Mean   :  33.760   Mean   :-19.47   Mean   :  -0.9385  
##  3rd Qu.: 517.0   3rd Qu.:  58.129   3rd Qu.: 16.12   3rd Qu.:  55.8335  
##  Max.   : 633.0   Max.   : 123.984   Max.   : 96.87   Max.   : 132.2337  
##                                                                          
##  total_accel_dumbbell gyros_dumbbell_x  gyros_dumbbell_y  gyros_dumbbell_z
##  Min.   : 1.0         Min.   :-1.0300   Min.   :-1.1100   Min.   :-1.180  
##  1st Qu.: 7.0         1st Qu.: 0.1600   1st Qu.:-0.2100   1st Qu.:-0.485  
##  Median :15.5         Median : 0.3600   Median : 0.0150   Median :-0.280  
##  Mean   :17.2         Mean   : 0.2690   Mean   : 0.0605   Mean   :-0.266  
##  3rd Qu.:29.0         3rd Qu.: 0.4625   3rd Qu.: 0.1450   3rd Qu.:-0.165  
##  Max.   :31.0         Max.   : 1.0600   Max.   : 1.9100   Max.   : 1.100  
##                                                                           
##  accel_dumbbell_x  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
##  Min.   :-159.00   Min.   :-30.00   Min.   :-221.0   Min.   :-576.0   
##  1st Qu.:-140.25   1st Qu.:  5.75   1st Qu.:-192.2   1st Qu.:-528.0   
##  Median : -19.00   Median : 71.50   Median :  -3.0   Median :-508.5   
##  Mean   : -47.60   Mean   : 70.55   Mean   : -60.0   Mean   :-304.2   
##  3rd Qu.:  15.75   3rd Qu.:151.25   3rd Qu.:  76.5   3rd Qu.:-317.0   
##  Max.   : 185.00   Max.   :166.00   Max.   : 100.0   Max.   : 523.0   
##                                                                       
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm    
##  Min.   :-558.0    Min.   :-164.00   Min.   :-176.00   Min.   :-63.500  
##  1st Qu.: 259.5    1st Qu.: -33.00   1st Qu.: -40.25   1st Qu.:-11.457  
##  Median : 316.0    Median :  49.50   Median :  94.20   Median :  8.830  
##  Mean   : 189.3    Mean   :  71.40   Mean   :  38.66   Mean   :  7.099  
##  3rd Qu.: 348.2    3rd Qu.:  96.25   3rd Qu.: 143.25   3rd Qu.: 28.500  
##  Max.   : 403.0    Max.   : 368.00   Max.   : 176.00   Max.   : 59.300  
##                                                                         
##   yaw_forearm       total_accel_forearm gyros_forearm_x  
##  Min.   :-168.000   Min.   :21.00       Min.   :-1.0600  
##  1st Qu.: -93.375   1st Qu.:24.00       1st Qu.:-0.5850  
##  Median : -19.250   Median :32.50       Median : 0.0200  
##  Mean   :   2.195   Mean   :32.05       Mean   :-0.0200  
##  3rd Qu.: 104.500   3rd Qu.:36.75       3rd Qu.: 0.2925  
##  Max.   : 159.000   Max.   :47.00       Max.   : 1.3800  
##                                                          
##  gyros_forearm_y   gyros_forearm_z   accel_forearm_x  accel_forearm_y 
##  Min.   :-5.9700   Min.   :-1.2600   Min.   :-212.0   Min.   :-331.0  
##  1st Qu.:-1.2875   1st Qu.:-0.0975   1st Qu.:-114.8   1st Qu.:   8.5  
##  Median : 0.0350   Median : 0.2300   Median :  86.0   Median : 138.0  
##  Mean   :-0.0415   Mean   : 0.2610   Mean   :  38.8   Mean   : 125.3  
##  3rd Qu.: 2.0475   3rd Qu.: 0.7625   3rd Qu.: 166.2   3rd Qu.: 268.0  
##  Max.   : 4.2600   Max.   : 1.8000   Max.   : 232.0   Max.   : 406.0  
##                                                                       
##  accel_forearm_z  magnet_forearm_x magnet_forearm_y magnet_forearm_z
##  Min.   :-282.0   Min.   :-714.0   Min.   :-787.0   Min.   :-32.0   
##  1st Qu.:-199.0   1st Qu.:-427.2   1st Qu.:-328.8   1st Qu.:275.2   
##  Median :-148.5   Median :-189.5   Median : 487.0   Median :491.5   
##  Mean   : -93.7   Mean   :-159.2   Mean   : 191.8   Mean   :460.2   
##  3rd Qu.: -31.0   3rd Qu.:  41.5   3rd Qu.: 720.8   3rd Qu.:661.5   
##  Max.   : 179.0   Max.   : 532.0   Max.   : 800.0   Max.   :884.0   
##                                                                     
##    problem_id   
##  Min.   : 1.00  
##  1st Qu.: 5.75  
##  Median :10.50  
##  Mean   :10.50  
##  3rd Qu.:15.25  
##  Max.   :20.00  
## 
```

The model is then tested with the testing dataset as follows:


```r
mytestingV2 <- mytestingV2[, -length(colnames(mytestingV2))]

newFrame <- head(training_dataset,1)
newFrame <- newFrame[, -length(colnames(newFrame))] 
fixedtestData <- rbind(newFrame, mytestingV2)
fixedtestData <- fixedtestData[-1,]


rfPredictTest <- predict(rfFit, fixedtestData, type="class")
rfPredictTest
```

```
##  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

Submission process
==================
The submission process is as follows:


```r
pml_write_files = function (x){
  n=length(x)
  for (i in 1:n){
    filename=paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}

pml_write_files(rfPredictTest)
```
