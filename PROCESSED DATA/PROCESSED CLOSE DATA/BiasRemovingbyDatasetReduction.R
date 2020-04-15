
#GETTING THE DATASET
setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA/PROCESSED CLOSE DATA")
credit = read.csv("BiasedClassifiable.csv")
credit$X = NULL

#NO. OF YES RECORDS THAT HAVE YES = 6636 , SO TAKING EQUAL NO OF RECORDS IN MNO
M1 = na.omit(credit[credit$Y == "1",])
M0 = na.omit(head(credit[credit$Y == "0",],674))
Mn1 = na.omit(head(credit[credit$Y == "-1",],674))
#CREATING THE MATRIX X WHICH HAS EQUAL YES'S AND NO'S. UNSHUFFLED
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