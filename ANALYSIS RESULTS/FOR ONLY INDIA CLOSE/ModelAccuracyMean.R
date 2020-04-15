setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/Analysis Results/FOR ONLY INDIA CLOSE/")
dataset = read.csv('Errors.csv', sep = '\t')
write.csv(dataset, 'ErrorData.csv')
X_MAE = (dataset$Mean.Absolute.Error / dataset$Mean.of.Test.Set)*100
Y_MAE = mean(X_MAE)
ModelAccuracyMean_MAE = 100-Y_MAE

Y_MSE = mean(dataset$Mean.Squared.Error)

X_RMSE = (dataset$Root.Mean.Squared.Error / dataset$Mean.of.Test.Set)*100
Y_RMSE = mean(X_RMSE)
ModelAccuracyMean_RMSE= 100-Y_RMSE




