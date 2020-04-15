
#GETTING THE DATASET
setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA/INDIA CLOSE OTHER OPEN")
credit = read.csv("Prev&CurrentDataset.csv", sep = '\t')
credit$X = NULL

M1 = na.omit(credit[credit$Class.2.. == "1",])
M0 = na.omit(head(credit[credit$Class.2.. == "0",],674))
Mn1 = na.omit(head(credit[credit$Class.2.. == "-1",],674))

x = rbind(M0  , M1)
x = rbind(x, Mn1)

#Check wether categorical data is factorized or not
#print(is.factor(x$SEX))
#print(is.factor(x$EDUCATION))
#print(is.factor(x$MARRIAGE))
#print(is.factor(x$AGE))

#SHUFFLING X
x <- x[sample(1:nrow(x)), ]
#View(x)
View(x)
x = na.omit(x)
write.csv(x,'unbiaseddataset.csv', sep = ',')