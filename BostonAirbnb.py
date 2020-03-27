import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re




bostondata = pd.read_csv("C:/NEU/Causal ML/Project/listings.csv")
print(bostondata.head())
print(bostondata.shape)
bostondata["price"] = bostondata["price"].map(lambda x: re.sub(r'\W+', '', x))

bostondata["price"]= bostondata["price"].astype(float) 

feature_with_na = [features for features in bostondata.columns if bostondata[features].isnull().sum()>1]

for feature in feature_with_na:
    print(feature, np.round(bostondata[feature].isnull().mean()*100,4), ' % missing values')

for feature in bostondata:
    bostondata[feature] = bostondata[feature].map(lambda x: re.sub(r'\W+', '', x))

print("thumbnails url {}".format(len(bostondata.thumbnail_url)))
print("medium url {}".format(len(bostondata.medium_url)))
print("xl picture  url {}".format(len(bostondata.xl_picture_url)))
print("neighbourhood group clensed {}".format(len(bostondata.neighbourhood_group_cleansed)))

print(bostondata.shape)


#numericals features
numerical_feature = [feature for feature in bostondata.columns if bostondata[feature].dtype != 'O']  
print('Number of numerical variables:', len(numerical_feature))  

categorical_feature = [feature for feature in bostondata.columns if bostondata[feature].dtype == 'O']  
print('Number of numerical variables:', len(categorical_feature))  
categorical_feature
#'O' is type object

for feature in categorical_feature:
    print('The feature is {} and number of categories are {}'.format(feature, len(bostondata[feature].unique())))

#year data
year_feature = [feature for feature in categorical_feature if 'last_scraped' in feature or 'host_since'  in feature or 'calendar_last_scraped' in feature or 'first_review' in feature or 'last_review'  in feature]
year_feature


#finding discrete features in numericals features
discrete_feature = [feature for feature in numerical_feature if len(bostondata[feature].unique())<25 and feature not in year_feature+['Id']]
discrete_feature
print('Number of discrete variables:', len(discrete_feature))  
bostondata[discrete_feature].head()


#finding continous features in numerical features
continous_feature = [feature for feature in numerical_feature if feature not in discrete_feature + year_feature + ['Id']]
print('Number of continous variables:', len(continous_feature))  
continous_feature

for feature in discrete_feature:
    data = bostondata.copy()
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(feature)
    plt.show()
    
    
for feature in continous_feature:
    data = bostondata.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(feature)
    plt.show()
    

#logarithmic transformation
for feature in continous_feature:
    data = bostondata.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['price'] = np.log(data['price']) 
        plt.scatter(data[feature],data['price'])
        plt.xlabel(feature)
        plt.ylabel('price')
        plt.title(feature)
        plt.show()


#outliers
for feature in continous_feature:
    data = bostondata.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column = feature)
        plt.xlabel(feature)
        plt.ylabel('price')
        plt.title(feature)
        plt.show()        
        

for feature in categorical_feature:
    data = bostondata.copy()
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(feature)
    plt.show()
    