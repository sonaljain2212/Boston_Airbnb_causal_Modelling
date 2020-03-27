import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import re

pd.pandas.set_option('display.max_columns',None)

#read the csv
bostondata = pd.read_csv('C:/NEU/Causal ML/Project/listings.csv')
bostondata.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bostondata, bostondata['price'],test_size =0.1, random_state = 0)

#removing $ sign and converting series data type into float
y_test = y_test.map(lambda x: re.sub(r'\W+', '', x))

y_train = y_train.map(lambda x: re.sub(r'\W+', '', x)).astype(float)


X_test['price'] = X_test['price'].map(lambda x: re.sub(r'\W+', '', x))
X_test['price'] = X_test['price'].astype(float)

X_train['price'] = X_train['price'].map(lambda x: re.sub(r'\W+', '', x))
X_train['price'] = X_train['price'].astype(float)


#categorical value with nan datatype denoted by object type 'O'
feature_with_nan = [features for features in X_train.columns if X_train[features].isnull().sum()>1 and X_train[features].dtype == 'O']
len(feature_with_nan)

for features in feature_with_nan:
    print('{}: {}% missing value'.format(features, np.around(X_train[features].isnull().mean()*100,4)))

#replacing missing values with a new label
def replace_cat_feature(dataset, features_nan):
    data = X_train.copy()
    data[feature_with_nan] = data[feature_with_nan].fillna('Missing')
    return data

X_train = replace_cat_feature(X_train, feature_with_nan)
X_train[feature_with_nan].isnull().sum()

#finding numerical features with missing values
numerical_with_nan = [features for features in X_train.columns if X_train[features].isnull().sum()>1 and X_train[features].dtype != 'O']
len(numerical_with_nan)
for features in numerical_with_nan:
    print('{}: {}% missing value'.format(features, np.around(X_train[features].isnull().mean()*100,4)))

#replacing nan values with median, 1 for missing and 0 for not missing

for feature in numerical_with_nan:
    median_value = X_train[feature].median()
    X_train[feature+'nan'] = np.where(X_train[feature].isnull(),1,0)
    X_train[feature].fillna(median_value, inplace= True)

X_train[numerical_with_nan].isnull().sum()

#temporal variables

#checkingzeros
numerical_with_zeros = [features for features in X_train[numerical_with_nan].columns if X_train[features].isin([0]).sum() ]

X_train[numerical_with_zeros]

num_features_withoutzeros = list(set(numerical_with_nan) - set(numerical_with_zeros))

#log normal distribution of numerical list without zeros
for features in num_features_withoutzeros:
    X_train[features] = np.log(X_train[features])
    
#finding categorical features
categorical_feature = [feature for feature in X_train.columns if X_train[feature].dtype == 'O']  
print('Number of numerical variables:', len(categorical_feature))  
categorical_feature

#% of each and every category considering the complete train dataset
'''
for features in categorical_feature:
    temp = X_train.groupby(features)['price'].count()/len(X_train)
    temp_df = temp[temp >0.01].index
    X_train[features] = np.where(X_train[features].isin(temp_df), X_train[features], 'Rare_var')     
'''   
#feature scaling

feature_scale = [feature for feature in X_train[numerical_with_nan].columns if feature not in ['id', 'price']]

len(feature_scale)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train[feature_scale])

data = pd.concat([X_train[['id','price']].reset_index(drop=True), pd.DataFrame(scaler.transform(X_train[feature_scale]), columns = feature_scale)], axis= 1)


data.head()

len(data)
