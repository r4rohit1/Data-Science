# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 23:04:06 2018

@author: rsudhakar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset=pd.read_table('file:///C:/Users/rsudhakar/Downloads/train_classifier.txt')
dataset.describe(include=['O'])
dataset.describe()
dataset.isnull().sum() 
dataset.info()
dataset.head(10)
dataset.tail(10)


#dataset["fruit_name"] = dataset["fruit_name"].map({'apple':0, 'mandrin':1,'orange':2,'lemon':3}).astype(int)

#used the following after google search as the above commented code gives error

#dataset[["fruit_label", "fruit_subtype","mass","width","height","color_score"]].sort_values(by='fruit_label')
print(dataset[['fruit_name','fruit_label']].groupby(['fruit_name']).count().sort_values(by='fruit_label', ascending = True))
print(dataset[['fruit_subtype','fruit_label']].groupby(['fruit_subtype']).count().sort_values(by='fruit_label', ascending = True))

g = sns.FacetGrid(dataset, col = 'fruit_name')
g.map(plt.hist, 'mass', bins=20)

grid = sns.FacetGrid(dataset, col='fruit_name', row='fruit_subtype', size=2.2, aspect=1.6)
grid.map(plt.hist, 'mass', alpha=.5, bins=20)
grid.add_legend()
# Business Presentation
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
newdata=le.fit(dataset['fruit_subtype'])
dataset['fruit_subtype']=le.transform(dataset['fruit_subtype'])

dataset.head(5)
dataset[]

Y=dataset['fruit_name']
X=dataset.drop('fruit_name', axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2)
help(train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round((logreg.score(X_test, Y_test) * 100), 2)
acc_log 


## Support Vector Machines/Classifiers
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc 

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn 

# GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gauss = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian 

# Perceptron is a special type of linear classifier, i.e. a classification algorithm that makes
# its predictions based on a linear predictor function
# combining a set of weights with the feature vector (Deep Learning, concept).
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_per = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron 

# Linear SVM
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc 

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree 

# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

#### Displaying the Training Accuracy of all the Models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


#Testing Model scores
acc_log_t=accuracy_score(Y_test,Y_pred_log)
acc_knn_t=accuracy_score(Y_test,Y_pred_knn)
acc_gaussian_t=accuracy_score(Y_test,Y_pred_gauss)
acc_perceptron_t=accuracy_score(Y_test,Y_pred_per)
acc_linear_svc_t=accuracy_score(Y_test,Y_pred_linear_svc)
acc_decision_tree_t=accuracy_score(Y_test,Y_pred_decision_tree)
acc_random_forest_t=accuracy_score(Y_test,Y_pred_random_forest)
acc_svc_t=accuracy_score(Y_test,Y_pred_svc)

models_test = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc_t, acc_knn_t, acc_log_t,
              acc_random_forest_t, acc_gaussian_t, acc_perceptron_t,
              acc_linear_svc_t, acc_decision_tree_t]})
models.sort_values(by='Score', ascending=False)


plt.xlabel('Algorithms')
plt.ylabel('Percentage')
width=1
height=10
result=[acc_log, acc_svc,acc_knn ,acc_gaussian,acc_perceptron,acc_linear_svc,acc_decision_tree,acc_random_forest ]
#y=['acc_log', 'acc_svc,acc_knn' ,'acc_gaussian,acc_perceptron','acc_linear_svc','acc_decision_tree','acc_random_forest' ]
plt.plot(result, color='blue')
plt.show()



