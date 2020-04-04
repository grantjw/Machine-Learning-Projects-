import pandas as pd
import numpy as np
pd.options.display.max_columns = 99

column_names =  ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 
        'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=column_names)

#Select all numeric columns, including dytype = object tht are numeric
cols_numeric = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 
                'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[cols_numeric]

numeric_cars = numeric_cars.replace('?',np.nan)
numeric_cars.head() 
numeric_cars.dtypes
numeric_cars = numeric_cars.astype(float)

#drop all Null values for price since price is target the variable 
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()

#replace all other null observations with column means 
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
numeric_cars.isnull().sum()

# Normalize scaling. 
# It is interesting to consider the affects of normalizing vs. standardizing data on model performance: 
# KNNs can perform better with normalization: https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
# There are two ways to normalize data that yield identical results.  
 
#1) using sklearn 
# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# numeric_cars = min_max_scaler.fit_transform(numeric_cars)

#2) using formulas 
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
def knn_train_test(train, target, df):
    np.random.seed(1)
    random_index = np.random.permutation(df.index)
    df_shuffled = df.reindex(random_index)
    # We use the 75/25 split, given that the data is not large, a 50/50 split may be worth considering 
    last_train_row = int(len(df_shuffled) * 3/4)
    train_df = df_shuffled.iloc[0:last_train_row]
    test_df = df_shuffled.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9,11,13,15]
    k_rmses = {}
    for k in k_values :
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train]],train_df[target])
        pred_var = knn.predict(test_df[[train]])
    
        mse = mean_squared_error(test_df[target], pred_var)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
    return k_rmses

rmse_results = {}

train_col = numeric_cars.columns.drop('price')

for i in train_col:
    rmse = knn_train_test(i, 'price', numeric_cars)
    rmse_results[i] = rmse

rmse_results

#plot k-value vs. RMSE 
import matplotlib.pyplot as plt

for k,v in rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x,y)
    plt.xlabel('k-value')
    plt.ylabel('RMSE')

####################### Multivariate Model with Hyperparameter tuning 
# compute average RMSE across different k for each feature 
feature_avg_rmse = {}
for k,v in rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) * 3/4)
    
    # We use the 75/25 split 
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

for nr_best_feats in range(2,7):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

k_rmse_results

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')
    
###
 



