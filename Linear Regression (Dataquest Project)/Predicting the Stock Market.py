
from IPython import get_ipython;   
get_ipython().magic('reset -sf')

import os
os.chdir("C:/Users/Lenovo 4/Desktop/Data Quest Folder/Predicting the Stock Market/")
os.getcwd()

import pandas as pd

stock = pd.read_table("sphist.csv", sep=",")

stock['Date'] = pd.to_datetime(stock['Date'])

# sort dataframe based on date

stock = stock.sort_values(by = ['Date'], ascending=True)

#set new index for reference 
stock['index'] = range(0,stock.shape[0],1)
stock.set_index(['index'])

# Generate a boolean series whether each item in the Date column is after 2015-04-01 

from datetime import datetime
stock['date_after_april1_2015'] = stock['Date'] > datetime(year=2015, month=4, day=1)



# Generate indicator: The average price from the past 5 days.
# 1) use iterrows option 

# stock.sort_values(by = ['Date'], ascending=False, inplace=True)
# wow = {}
# for index, row in stock.iterrows():
#     five_day = stock['Close'].iloc[index+1 : index+6].mean()
#     wow[index] = five_day

# stock['day_5'] = pd.Series(wow)
# stock['day_5'].iloc[16585:] = 0 


# 2) Use pandas rolling option 
stock['day_mean_5'] = stock['Close'].rolling(window=5).mean().shift(1, axis=0)

# The average price from the past 365 days.
stock['day_mean_365'] = stock['Close'].rolling(window=365).mean().shift(1, axis=0)

# The ratio between the average price for the past 5 days, and the average price for the past 365 days.
stock['ratio_5_365'] = stock['day_mean_5'] / stock['day_mean_365']

# The standard deviation of the price over the past 5 days.
stock['day_sd_5'] = stock['Close'].rolling(window=5).std().shift(1, axis=0)

# The standard deviation of the price over the past 365 days.
stock['day_sd_365'] = stock['Close'].rolling(window=365).std().shift(1, axis=0)

# The ratio between the standard deviation for the past 5 days, and the standard deviation for the past 365 days.
stock['ratio_5_365'] = stock['day_sd_5'] / stock['day_sd_365']


stock = stock[stock["Date"] > datetime(year=1951, month=1, day=2)]
stock = stock.dropna(axis=0)
train = stock[stock["Date"] < datetime(year=2013, month=1, day=1)]
test = stock[stock["Date"] >= datetime(year=2013, month=1, day=2)]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
features = ['day_mean_5', 'day_mean_365', 'ratio_5_365', 'day_sd_5', 'day_sd_365', 'ratio_5_365']
train_features = train[features]
test_features = test[features]
lr.fit(train_features,train['Close'])

predicted_labels = lr.predict(test_features)

#mae
import numpy as np
test['error'] = np.absolute(predicted_labels - test['Close'])
mae = test['error'].mean()
print(mae)




