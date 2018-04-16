# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:16:18 2018

@author: rsudhakar
"""

import pandas as pd
import numpy as np
import re

# importing dataset
train=pd.read_csv('train_E6oV3lV.csv')
test=pd.read_csv('test_tweets_anuFYb8.csv')


#import nltk
#nltk.download('stopwords')
count=0
corpus=[]
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
for i in range(0,31963):
    review=re.sub('[^a-zA-z]',' ',train['tweet'][i])
    review.lower()
    review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word  in review ]
    review=''.join(review)
    corpus.append(review)
    count+=1
    len(corpus)
    print("The sequence number finsihed is:",count ,"\nThe count of corpus is :", len(corpus))
    
corpus_t=[]
for i in range(0,17197):
    review_t=re.sub('[^a-zA-z]',' ',test['tweet'][i])
    review_t.lower()
    review_t.split()
    ps=PorterStemmer()
    review_t=[ps.stem(word) for word  in review_t ]
    review_t=''.join(review_t)
    corpus_t.append(review_t)
    count+=1
    len(corpus_t)
    print("The sequence number finsihed is:",count ,"\nThe count of corpus is :", len(corpus_t))

from sklearn.feature_extraction.text import CountVectorizer
cs=CountVectorizer( stop_words='english')
X=cs.fit_transform(corpus).toarray()
Y=train['label']
T=cs.fit_transform(corpus_t).toarray()





    
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X, Y)
Y_pred_log = logreg.predict(T)
acc_log = round(logreg.score(X, Y) * 100, 2)
acc_log ### Accuracy, we have, here is ~83.24


## Support Vector Machines/Classifiers
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc ### Accuracy, we have, here is ~89.75

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn ### Accuracy, we have, here is ~84.55

# GaussianNB
gaussian = GaussianNB()
gaussian.fit(X, Y)
Y_pred_gauss = gaussian.predict(T)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian ### Accuracy, we have, here is ~78.65

# Perceptron is a special type of linear classifier, i.e. a classification algorithm that makes
# its predictions based on a linear predictor function
# combining a set of weights with the feature vector (Deep Learning, concept).
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_per = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron ### Accuracy, we have, here is ~73.88

# Linear SVM
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc ### Accuracy, we have, here is ~62.92

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree ### Accuracy, we have, here is ~98.3, High Accuracy, but may be Overfitting

# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest ### Accuracy, we have, here is ~98.31

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

# Calculating the accuracy on the test data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

## Logistic Regression
report_log = classification_report(Y_test, Y_pred_log)
print(report_log)
matrix_log = confusion_matrix(Y_test, Y_pred_log)
print(matrix_log)
accuracy_log = accuracy_score(Y_test, Y_pred_log)
print(accuracy_log)

## Support Vector Machines/Classifiers
report_svc = classification_report(Y_test, Y_pred_svc)
print(report_svc)
matrix_svc = confusion_matrix(Y_test, Y_pred_svc)
print(matrix_svc)
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
print(accuracy_svc)

# KNeighborsClassifier
report_knn = classification_report(Y_test, Y_pred_knn)
print(report_knn)
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
print(matrix_knn)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print(accuracy_knn)

# GaussianNB
report_gauss = classification_report(Y_test, Y_pred_gauss)
print(report_gauss)
matrix_gauss = confusion_matrix(Y_test, Y_pred_gauss)
print(matrix_gauss)
accuracy_gauss = accuracy_score(Y_test, Y_pred_gauss)
print(accuracy_gauss)

Perceptron
report_per = classification_report(Y_test, Y_pred_per)
print(report_per)
matrix_per = confusion_matrix(Y_test, Y_pred_per)
print(matrix_per)
accuracy_per = accuracy_score(Y_test, Y_pred_per)
print(accuracy_per)

# Linear SVM
report_linear_svc = classification_report(Y_test, Y_pred_linear_svc)
print(report_linear_svc)
matrix_linear_svc = confusion_matrix(Y_test, Y_pred_linear_svc)
print(matrix_linear_svc)
accuracy_linear_svc = accuracy_score(Y_test, Y_pred_linear_svc)
print(accuracy_linear_svc)

# Decision Tree Classifier
report_decision_tree = classification_report(Y_test, Y_pred_decision_tree)
print(report_decision_tree)
matrix_decision_tree = confusion_matrix(Y_test, Y_pred_decision_tree)
print(matrix_decision_tree)
accuracy_decision_tree = accuracy_score(Y_test, Y_pred_decision_tree)
print(accuracy_decision_tree)

# Random Forest Classifier
report_random_forest = classification_report(Y_test, Y_pred_random_forest)
print(report_random_forest)
matrix_random_forest = confusion_matrix(Y_test, Y_pred_random_forest)
print(matrix_random_forest)
accuracy_random_forest = accuracy_score(Y_test, Y_pred_random_forest)
print(accuracy_random_forest)

### Test Accuracy of the Models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Linear SVC',
              'Decision Tree'],
    'Score': [accuracy_svc, accuracy_knn, accuracy_log,
              accuracy_random_forest, accuracy_gauss, accuracy_per,
              accuracy_linear_svc, accuracy_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

