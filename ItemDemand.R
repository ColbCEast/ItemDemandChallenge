# Run for ALL Models

setwd("C:/Users/colby/OneDrive/Desktop/STAT 348/ItemDemandChallenge")

library(tidymodels)
library(timetk)
library(vroom)

train_data <- vroom("train.csv")
view(train_data)
test_data <- vroom("test.csv")

# Exploratory Data Analysis