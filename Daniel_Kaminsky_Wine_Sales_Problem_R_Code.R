### Daniel Kaminsky - Wine Sales Problem ###
### Kaggle competition name: DanielK ###

# In case they are not installed, install the following packages
install.packages(c("aod", "car", "caTools", "flux", "ggplot2", "grid", "gridExtra", "gvlma", "leaps",
                   "MASS", "Metrics", "moments", "plyr", "pscl", "rattle", "ROCR", "rpart", "scales"))
# Load the packages 
library(aod)
library(car)
library(caTools)
library(flux)
library(ggplot2)
library(grid)
library(gridExtra)
library(gvlma)
library(leaps)
library(MASS)
library(Metrics)
library(moments)
library(plyr)
library(pscl) # For Zero Inflated Poisson and Negative Binomial Regressions
library(rattle)
library(ROCR)
library(rpart)
library(rpart.plot)
library(scales)


## Creating the TRAINING Dataset ###
# Reading the file into R
winedata <- read.csv("D:/wine.csv", sep = ",")

# Check winedata using str()
str(winedata) # 'data.frame': 12795 obs. of 16 variables
head(winedata)

# Dropping the INDEX so I can minimize the effort of writting all the
# variables each time in each regression model
mydata <- winedata[c(-1)]

# Check mydata using str()
str(mydata) # 'data.frame': 12795 obs. of 15 variables
head(mydata)

# Missing Values - Count per Variable
sapply(mydata, function(mydata) sum(is.na(mydata)))

# Fixing Missing Values with MEDIANs. Select rows where the Variable Observation is NA and replace it with MEDIAN
mydata$ResidualSugar[is.na(mydata$ResidualSugar)==T] <- median(mydata$ResidualSugar, na.rm = TRUE)
median(mydata$ResidualSugar, na.rm = F) # Testing for NA values
mydata$Chlorides[is.na(mydata$Chlorides)==T] <- median(mydata$Chlorides, na.rm = TRUE)
median(mydata$Chlorides, na.rm = F) # Testing for NA values
mydata$FreeSulfurDioxide[is.na(mydata$FreeSulfurDioxide)==T] <- median(mydata$FreeSulfurDioxide, na.rm = TRUE)
median(mydata$FreeSulfurDioxide, na.rm = F) # Testing for NA values
mydata$TotalSulfurDioxide[is.na(mydata$TotalSulfurDioxide)==T] <- median(mydata$TotalSulfurDioxide, na.rm = TRUE)
median(mydata$TotalSulfurDioxide, na.rm = F) # Testing for NA values
mydata$pH[is.na(mydata$pH)==T] <- median(mydata$pH, na.rm = TRUE)
median(mydata$pH, na.rm = F) # Testing for NA values
mydata$Sulphates[is.na(mydata$Sulphates)==T] <- median(mydata$Sulphates, na.rm = TRUE)
median(mydata$Sulphates, na.rm = F) # Testing for NA values
mydata$Alcohol[is.na(mydata$Alcohol)==T] <- median(mydata$Alcohol, na.rm = TRUE)
median(mydata$Alcohol, na.rm = F) # Testing for NA values
mydata$STARS[is.na(mydata$STARS)==T] <- median(mydata$STARS, na.rm = TRUE)
median(mydata$STARS, na.rm = F) # Testing for NA values

# Missing Values - Count per Variable
sapply(mydata, function(mydata) sum(is.na(mydata)))

# Use summary() to obtain and present descriptive statistics from mydata.
summary(mydata)

# If a value in column "Alcohol" is Less than 0 set it to 0
mydata[mydata$Alcohol < 0, "Alcohol"] = 0
summary(mydata)

# Histogram and Q-Q Plots Alcohol
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(mydata$Alcohol, col = "deepskyblue3", main = "Histogram of Alcohol", xlab = "Alcohol",
     cex = 2, cex.axis = 1.5, cex.lab = 2.0, cex.main = 2, cex.sub = 1.5)
qqnorm(mydata$Alcohol, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 2.0, cex.main = 2, cex.sub = 1.5)
qqline(mydata$Alcohol, col = "darkred", lty = 2, lwd = 3)
boxplot(mydata$Alcohol[mydata$Alcohol], col = "red", ylim = c(0.00, 2000), pch = 16,
        main = "Alcohol", cex = 2.0, cex.axis = 1.65, cex.lab = 1.75, cex.main = 2.0)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Model 0 - Base Model Poisson Regression
summary(Model0 <- glm(TARGET ~  ., family="poisson", data=mydata))

# Model 1 - Poisson Regression - Stepwise Variable Selection
Model1 <- glm(TARGET ~  ., family="poisson", data=mydata)
summary(VarSelection <- step(Model1, direction="both"))

# Model 2 - Negative Binomial  Regression - Stepwise Variable Selection
Model2 <- glm.nb(TARGET ~  ., data=mydata)
summary(VarSelectionNB <- step(Model2, direction="both"))

# Model 3 - Zero Inflated Poisson Regression
summary(Model3 <- zeroinfl(TARGET ~  ., data=mydata))

# Using the Vuong Test to Compare the Poisson Model 1 and the Zero Inflated Poisson Model 3
vuong(Model3, VarSelection)
vuong(Model3, Model1)
# The Vuong test compares the zero-inflated model with an ordinary Poisson regression model.
# In this example, we can see that our test statistic is significant, indicating that the
# zero-inflated model is superior to the standard Poisson model.

# Model 4 - Zero Inflated Negative Binomial Regression
Model4 <- zeroinfl(TARGET ~  .,
                   data=mydata, dist = "negbin", EM = TRUE)
summary(Model4)
vuong(Model3, Model4)

# Model 5 - MLR
summary(Model5 <- lm(TARGET ~  ., data=mydata))

# Model 6 - MLR with Stepwise Variable Selection
summary(Model6 <- lm(TARGET ~  ., data=mydata))
summary(VarSelectionMLR <- step(Model6, direction="both"))

# Preparing for RMSE metric
Actual <- mydata$TARGET
Pred_Model0 <- fitted(Model0)
Pred_Model1 <- fitted(Model1)
Pred_VarSelection <- fitted(VarSelection)
Pred_Model2 <- fitted(Model2)
Pred_Model3 <- fitted(Model3)
Pred_Model4 <- fitted(Model4)
Pred_Model5 <- fitted(Model5)
Pred_VarSelectionMLR <- fitted(VarSelectionMLR)

# RMSE library(Metrics)
rmse(Actual, Pred_Model0)
rmse(Actual, Pred_Model1)
rmse(Actual, Pred_VarSelection)
rmse(Actual, Pred_Model2)
rmse(Actual, Pred_Model3)
rmse(Actual, Pred_Model4)
rmse(Actual, Pred_Model5)
rmse(Actual, Pred_VarSelectionMLR)

# Decision Tree of All Variables with rattle()
rattle()

#### CREATING A TEST DATASET ###
# Reading the file into R
TESTdata <- read.csv("D:/wine_test.csv", sep = ",")

# Check mydata using str()
str(TESTdata) # 'data.frame': 3335 obs. of 16 variables
head(TESTdata)

# Missing Values - Count per Variable
sapply(TESTdata, function(TESTdata) sum(is.na(TESTdata)))

# Fixing Missing Values with MEDIANs. Select rows where the Variable Observation is NA and replace it with MEDIAN
TESTdata$ResidualSugar[is.na(TESTdata$ResidualSugar)==T] <- median(TESTdata$ResidualSugar, na.rm = TRUE)
median(TESTdata$ResidualSugar, na.rm = F) # Testing for NA values
TESTdata$Chlorides[is.na(TESTdata$Chlorides)==T] <- median(TESTdata$Chlorides, na.rm = TRUE)
median(TESTdata$Chlorides, na.rm = F) # Testing for NA values
TESTdata$FreeSulfurDioxide[is.na(TESTdata$FreeSulfurDioxide)==T] <- median(TESTdata$FreeSulfurDioxide, na.rm = TRUE)
median(TESTdata$FreeSulfurDioxide, na.rm = F) # Testing for NA values
TESTdata$TotalSulfurDioxide[is.na(TESTdata$TotalSulfurDioxide)==T] <- median(TESTdata$TotalSulfurDioxide, na.rm = TRUE)
median(TESTdata$TotalSulfurDioxide, na.rm = F) # Testing for NA values
TESTdata$pH[is.na(TESTdata$pH)==T] <- median(TESTdata$pH, na.rm = TRUE)
median(TESTdata$pH, na.rm = F) # Testing for NA values
TESTdata$Sulphates[is.na(TESTdata$Sulphates)==T] <- median(TESTdata$Sulphates, na.rm = TRUE)
median(TESTdata$Sulphates, na.rm = F) # Testing for NA values
TESTdata$Alcohol[is.na(TESTdata$Alcohol)==T] <- median(TESTdata$Alcohol, na.rm = TRUE)
median(TESTdata$Alcohol, na.rm = F) # Testing for NA values
TESTdata$STARS[is.na(TESTdata$STARS)==T] <- median(TESTdata$STARS, na.rm = TRUE)
median(TESTdata$STARS, na.rm = F) # Testing for NA values

# Missing Values - Count per Variable
sapply(TESTdata, function(TESTdata) sum(is.na(TESTdata)))

# Use summary() to obtain and present descriptive statistics from mydata.
summary(TESTdata)

# If a value in column "Alcohol" is Less than 0 set it to 0
TESTdata[TESTdata$Alcohol < 0, "Alcohol"] = 0
summary(TESTdata)

# Regression Model Test Dataset
TESTdata$LN_TARGET <-	(  -5.7157438
                         +TESTdata$FixedAcidity *	0.0032405
                         +TESTdata$VolatileAcidity *	0.2436976
                         +TESTdata$CitricAcid *	-0.0824209
                         +TESTdata$ResidualSugar *	-0.0013965
                         +TESTdata$Chlorides *	0.2703606
                         +TESTdata$FreeSulfurDioxide *	-0.0007122
                         +TESTdata$TotalSulfurDioxide *	-0.0008634
                         +TESTdata$Density *	0.8737988
                         +TESTdata$pH *	0.1960018
                         +TESTdata$Sulphates *	0.1208217
                         +TESTdata$Alcohol *	0.0101778
                         +TESTdata$LabelAppeal *	0.3264473
                         +TESTdata$AcidIndex *	0.4845997
                         +TESTdata$STARS *	-0.6933846)

# Checking TESTdata
head(TESTdata)
tail(TESTdata)

# Transforming to Probabilities by Exponentiating the LN_TARGET variable
#with the natural exponent ("e")
TESTdata$P_TARGET <- (exp(TESTdata$LN_TARGET))

# Use summary() to obtain and present descriptive statistics from mydata.
str(TESTdata)
summary(TESTdata)
head(TESTdata)
tail(TESTdata)

#### ### ### ### ### ### ### ### ### ### ### ### ###
### Creating the Output file with the INDEX, P_TARGET
Daniel_Kaminsky_WineSales_Sec60_R_Output <- data.frame(TESTdata$ï..INDEX, TESTdata$P_TARGET)
names(Daniel_Kaminsky_WineSales_Sec60_R_Output) <-c("INDEX", "P_TARGET")

# Checking the Output Dataset
str(Daniel_Kaminsky_WineSales_Sec60_R_Output) # 'data.frame': 3335 obs. of 2 variables
head(Daniel_Kaminsky_WineSales_Sec60_R_Output)
tail(Daniel_Kaminsky_WineSales_Sec60_R_Output)

#### ### ### ### ### ### ### ### ### ### ### ### ###
### Write TESTdata to CSV ###
write.csv(Daniel_Kaminsky_WineSales_Sec60_R_Output, 
          file = "D:/WineSales_Output.csv", 
          row.names = FALSE)















