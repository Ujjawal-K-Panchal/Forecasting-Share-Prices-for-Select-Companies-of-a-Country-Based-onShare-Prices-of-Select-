setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/UNPROCESSED DATA/UNPROCESSED US DATA")
fileNames <- Sys.glob("*.csv")
for (fileName in fileNames) {
  
  # read data:
  dataset <- read.csv(fileName, skip = 1)
  dataset$CLOSE = dataset$Close
  
  proc = data.frame( dataset$Date , dataset$CLOSE)
  proc$date = proc$dataset.ï..Date
  proc$dataset.ï..Date = NULL
  
  # add more stuff here
  setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA/New folder/")
  write.csv(proc , fileName)
  setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/UNPROCESSED DATA/UNPROCESSED US DATA")
}
#dataset = read.csv('Container Corporation of India.csv', skip = 1)
#View(dataset)
#dataset$OHLC_avg = (dataset$Open + dataset$High + dataset$Low + dataset$Close)/4

#proc = data.frame( dataset$Date , dataset$OHLC_avg)
#proc$date = proc$dataset.ï..Date
#proc$dataset.ï..Date = NULL
#View(proc)
