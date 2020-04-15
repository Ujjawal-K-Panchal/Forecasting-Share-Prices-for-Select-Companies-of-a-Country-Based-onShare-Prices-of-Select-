setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA")
dataset = read.csv('ForIndia65.csv' , na.strings = 'NA' , na.omit(TRUE))
library("caTools", lib.loc="~/R/win-library/3.4")
set.seed(7)
split = sample.split(dataset$India.65 , SplitRatio = 0.7)
training = subset(dataset , split ==TRUE)
test = subset(dataset , split == FALSE)
View(dataset)
training$dataset.Date = NULL
test$dataset.Date = NULL
training$X = NULL
test$X = NULL
model = lm(India.65~. , training , family = binomial )
prediction = predict(model, newdata = test , type = "response")
summary(model)