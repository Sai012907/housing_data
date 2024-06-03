import csv
import pandas as pd
import numpy as np

housing = pd.read_csv("data.csv")

from pandas.plotting import scatter_matrix

attributes = ["price", "floor_size", "number_of_bathrooms", "number_of_bedrooms"]
scatter_matrix(housing[attributes])

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)

housing = housing.drop("address", axis = 'columns')
housing = housing.drop("residence_type", axis = 'columns')
housing = housing.drop("street_address", axis = 'columns')
housing = housing.drop("house_type", axis = 'columns')
housing = housing.drop("postal_code", axis = 'columns')

def process_price(row):
    ireturn = 0
    i = row['price']
    ireturn = np.sqrt(i)
    return ireturn
    
#housing['price'] = housing.apply(process_price, axis = 1)

def process_pool(row):
    
    ireturn = 0
    i = row['pool']
    
    if i == "No":
        ireturn = 0
    elif i == "community-pool":
        ireturn = 1
    else:
        ireturn = 2
        
    return ireturn

housing['pool'] = housing.apply(process_pool, axis = 1)

def process_bathrooms(row):
    
    ireturn = 0
    i = row['number_of_bathrooms']
    
    try:
        ireturn = int(i[0]) + 0.5*int(i[4])
    except:
        try:
            ireturn = int(i)
        except:
            ireturn = None
            
    return ireturn

housing['number_of_bathrooms'] = housing.apply(process_bathrooms, axis = 1)

def process_year_built(row):
    
    ireturn = 0
    i = row['year_built']
    
    try:
        ireturn = int(i[0]) + 2023-int(i)
    except:
        ireturn = None
        
    return ireturn

housing['year_built'] = housing.apply(process_year_built, axis = 1)

from haversine import haversine

def process_distance(housing):

    greater_boston = (42.3611, -71.0571)

    coordinates = []
    latitude = housing["latitude"]
    longitude = housing["longitude"]
    distances = []

    for i in range(0, len(latitude)):
        try:
            coordinates.append((latitude[i], longitude[i]))
        except:
            coordinates.append(None)
        
    for coord in coordinates:
        if coord:
            dist = haversine(greater_boston, coord, unit = 'mi')
            distances.append(dist)
        else:
            distances.append(None)
        
    distances = pd.DataFrame(distances)

    housing['distance_to_greater_boston'] = distances
    
process_distance(housing)

housing = pd.get_dummies(housing)

from scipy import stats
    
housing = housing.dropna()

q1 = housing["price"].quantile(0.25)
q3 = housing["price"].quantile(0.75)
iqr = q3 - q1

outliers = housing[(housing["price"] < (q1 - 1.5 * iqr)) | (housing["price"] > (q3 + 1.5 * iqr))]
housing = housing.drop(outliers.index)

housing = housing.dropna()

housing = housing.reset_index(drop=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics

X = housing.drop("price", axis = 'columns')
y = housing["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 31, test_size = 0.1, shuffle=True)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))

fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test,'r')

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'polynomialfeatures__degree': [2, 3, 4, 5],
    'lasso__alpha': [5000, 6000, 7000, 8000]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomialfeatures', PolynomialFeatures()),
    ('lasso', Lasso()),
])

grid_search = GridSearchCV(pipeline, param_grid, cv = 3, scoring = 'neg_mean_squared_error')
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)           

print(metrics.mean_squared_error(y_test, y_pred, squared = False))  

import pickle

pickle.dump(grid_search, open('model.pkl','wb'))
