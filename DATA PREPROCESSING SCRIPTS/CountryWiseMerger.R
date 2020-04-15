#cbindPad is a custom made function from stackoverflow forums to bind two datasets with diff no. of columns and add NAs to the shorter one.
cbindPad <- function(...){
  args <- list(...)
  n <- sapply(args,nrow)
  mx <- max(n)
  pad <- function(x, mx){
    if (nrow(x) < mx){
      nms <- colnames(x)
      padTemp <- matrix(NA, mx - nrow(x), ncol(x))
      colnames(padTemp) <- nms
      if (ncol(x)==0) {
        return(padTemp)
      } else {
        return(rbind(x,padTemp))
      }
    }
    else{
      return(x)
    }
  }
  rs <- lapply(args,pad,mx)
  return(do.call(cbind,rs))
}
#Script Beginning:
setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA")
dataset = read.csv('3M.csv')
setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA/PROCESSED US DATA")
fileNames <- Sys.glob("*.csv")
for(fileName in fileNames)
{
  data = read.csv(fileName)
  data$X = NULL
  data$dataset.Date = NULL
  dataset = cbindPad(dataset,data)
  
}
setwd("C:/Users/uchih/Desktop/Internship IIM Data/India-Japan-US-StockPriceData/PROCESSED DATA")
write.csv(dataset, 'MergedUS.csv')