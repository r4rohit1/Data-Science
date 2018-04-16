import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Reading data from the data files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Exploratory Data Analysis
print(train.columns.values)
print(test.columns.values)
train.head(10)
train.info()
train.describe()
train.describe(include=['O'])

# Aggregating Values by Columns
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).count().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).count().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).count().sort_values(by='Survived', ascending=False)

# Visualisation, plotting graphs on the Grid
g = sns.FacetGrid(train, col = 'Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
train.head()

## Feature Engineering: Creation of new Columns
# Creating Age Bands for Age and Analysis
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

## Calulation of Family Size: No. of Siblings + Parents/Children + 1 and Analysis
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

## Determining, if the passenger was travelling alone or not, and Analysis
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#### Imputation: Filling in missing Values
# Imputation for Embarked done with the value having max count
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)
## Mapping the Embarked with Int Values(Required for Logistic Regression)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train['Sex'] = train['Sex'].map({'female':1, 'male':0}).astype(int)
train['Age'] = train['Age'].fillna(train['Age'].median())

train.head()
train.describe()

### Dropping the non-essential columns from the train dataset
non_essential_columns = ['PassengerId','Name','Cabin','AgeBand','Ticket']
for cols in non_essential_columns:
    train = train.drop(cols, axis = 1)


## Checking the data now
print(train.head())

#### Splitting up the data, preparing the data to be fed to the ML Algos

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(train, test_size=0.2)

X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]

X_test = test_data.drop("Survived", axis=1)
Y_test = test_data["Survived"]

####### Machine Learning Algos implementation start here #########
### First we will see the accuracy on the training Data, and then on the test data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
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
gaussian.fit(X_train, Y_train)
Y_pred_gauss = gaussian.predict(X_test)
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