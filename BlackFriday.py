# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:03:07 2018

@author: rsudhakar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

#import the train data
train=pd.read_csv('train.csv')

#preprocesing training data
#train.describe()
#train.info()
#train['Product_Category_2'].head()
#train['Product_Category_3'].isnull().sum()

#train['Product_Category_2'] = train['Product_Category_2'].fillna(train['Product_Category_2'].min())
#train['Product_Category_3'] = train['Product_Category_3'].fillna(train['Product_Category_3'].min())
train[train.dtypes[(train.dtypes=="float64")|(train.dtypes=="int64")]
                       .index.values].hist(figsize=[11,11])
#train['Product_Category_2'].describe()

del train['Product_Category_2']
del train['Product_Category_3']
cutoff_purchase = np.percentile(train['Purchase'], 99.9)  # 99.9 percentile
train.ix[train['Purchase'] > cutoff_purchase, 'Purchase'] = cutoff_purchase

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
train['Gender']=le.fit_transform(train['Gender'])
train['City_Category']=le.fit_transform(train['City_Category'])
train['User_ID']=le.fit_transform(train['User_ID'])
train['Product_ID']=le.fit_transform(train['Product_ID'])
train['Stay_In_Current_City_Years']=le.fit_transform(train['Stay_In_Current_City_Years'])
train['Age']=le.fit_transform(train['Age'])

#import the test data
test=pd.read_csv('test.csv')

#preprocesing Test data
#test['Product_Category_2'] = test['Product_Category_2'].fillna(test['Product_Category_2'].min())
#test['Product_Category_3'] = test['Product_Category_3'].fillna(test['Product_Category_3'].min())


test['Gender']=le.fit_transform(test['Gender'])
test['City_Category']=le.fit_transform(test['City_Category'])
test['User_ID']=le.fit_transform(test['User_ID'])
test['Product_ID']=le.fit_transform(test['Product_ID'])
test['Stay_In_Current_City_Years']=le.fit_transform(test['Stay_In_Current_City_Years'])
test['Age']=le.fit_transform(test['Age'])


del test['Product_Category_2']
del test['Product_Category_3']
# Splitting data into predcitor
Y_train=train['Purchase']
X_train= train.drop("Purchase", axis=1)

#Model building
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=11, random_state=0,min_samples_leaf=7)
regr.fit(X_train,Y_train)
Y_pred_regr=regr.predict(test)

text=np.array(Y_pred_regr)
df = pd.DataFrame(columns=['User_ID','Product_ID ','Purchase'])


test_original=pd.read_csv('test.csv')

df['User_ID']=test_original['User_ID']
df['Product_ID']=test_original['Product_ID']
df['Purchase']=pd.DataFrame(text)

df.to_csv("SampleSubmission.csv",index= False)  







