rm(list=ls())

setwd("/Users/grant/Desktop/Desktop/Poli 5D/Rwork/Project/")

## Section 1 Datasets Merged 

library(dplyr)
library(tidyr)
GDPdata <- read.csv("GDP.csv")
KOFdata <- read.csv("KOF_data.csv")
CDPdata <- read.csv("CDPS.csv")
KOFdata <- KOFdata[,c(1,2,3,12,15)]
GDPdata <- GDPdata[,c(3,4,5,7)]
CDPdata <- CDPdata[,c("country", "year", "labfopar", "iso")]
CDPdata <- rename(CDPdata, code = iso)
KOFdata$country <- gsub("Slovak Republic", "Slovakia", KOFdata$country)
KOFdata$country <- gsub("United States", "USA", KOFdata$country)
c
GDPdata <-rename(GDPdata, country = Country.Name , year = Time, code = Country.Code, GDP = Value)


data <- inner_join(CDPdata,KOFdata, c("year", "code"))
data <- rename(data, fin = KOFFiGIdj, trade = KOFTrGIdj)
data <- inner_join(data, GDPdata, c("year","code"))
data <- data[, c(2,3,4,6,7,8,9)]



## Section 3 Panel Data Regression ## 

library(plm)

##Lag financial globalization 
panel.data <- pdata.frame(data, index=c("country", "year"))
panel.data$lag.fin <- lag(panel.data$fin, 1)

## NA omit
panel.data <- na.omit(panel.data)
##Panel data with both entity and time fixed effects
##Control: labor pariticpation and trade 
mod.panel <- plm(GDP ~ lag.fin + trade + labfopar,   
           data = panel.data,
           index = c("country", "year"), 
           model = "within", 
           effect = "twoways")
summary(mod.panel)

## Section 2 Outliers Removed using boxplot
outliers.GDP <- boxplot(panel.data$GDP)$out
outliers.lagfin <- boxplot(panel.data$lag.fin)$out 
panel.data.removed <- panel.data[-which(panel.data$GDP %in% outliers.GDP),]
panel.data.removed <- panel.data.removed[-which(panel.data.removed$lag.fin %in% outliers.lagfin),]
print(outliers.lagfin)
print(outliers.GDP)

## Section 3 Panel Data Regression with Removed Outliers Fixed country and time effects 

mod.panel.removed <- plm(GDP ~ lag.fin + trade + labfopar,   
           data = panel.data.removed,
           index = c("country", "year"), 
           model = "within", 
           effect = "twoways")
summary(mod.panel.removed)

## SECTION H ## Histogram for IV and DV 
hist(panel.data.removed$GDP, freq=FALSE,
     xlab= "GDP per capita, PPP (constant 2011 international $)",
     ylab= "probability densities",
     main = "Distribution of GDP per capita, PPP (constant 2011 international $) in probability densities")

## SECTION H ## Histogram for lag.fin
hist(panel.data.removed$lag.fin, freq=FALSE,
     xlab= "Financial Globalization Indicator",
     ylab= "probability densities",
     main = "Distribution of financial globalization in probability densities")




## SECTION 4 Partial Regression plot outliers exist 
fin.mod <- lm(lag.fin ~ trade + labfopar + country + year, data = panel.data)
x1.hat <-predict(fin.mod)

gdp.mod <- lm(GDP ~ trade + labfopar + country + year, data = panel.data)
y1.hat <-predict(gdp.mod)
mod1 <- lm(y1.hat ~ x1.hat)

plot(y1.hat ~ x1.hat, xlab= "RES:X1 VERSUS OHTER X", ylab= "RES:X1 REMOVED",
     main="Partial Regression Plot with Outliers") 
abline(mod1)
summary(mod1)

## SECTION 4 Partial Regression plot outliers removed 
fin.mod.removed <- lm(lag.fin ~ trade + labfopar + country + year, data = panel.data.removed)
x1.hat.removed <-predict(fin.mod.removed)

gdp.mod.removed <- lm(GDP ~ trade + labfopar + country + year, data = panel.data.removed)
y1.hat.removed <-predict(gdp.mod.removed)
mod1.removed <- lm(y1.hat.removed ~ x1.hat.removed)

plot(y1.hat.removed ~ x1.hat.removed, xlab= "RES:X1 VERSUS OHTER X", ylab= "RES:X1 REMOVED",
     main="Partial Regression Plot Outliers Removed") 

abline(mod1.removed)
summary(mod1.removed)



## ADDITIONAL checks for extreme outliers 
car::outlierTest(mod1.removed)

##unique country numbers
country.numbers <- unique(panel.data$country)
length(country.numbers)