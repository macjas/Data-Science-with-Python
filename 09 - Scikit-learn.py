import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy import stats
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#############################################################################################################
# 1. Getting started with scikit-learn
############################################################################################################# 
from sklearn import datasets
iris = datasets.load_iris()      
digits = datasets.load_digits()

# About scikit-learn datasets: bunch objects
iris.keys()         # see whats available in a dataset
type(iris)          # the scikit datasets data are stored as form of array
print iris.data     # independent variables are stored as .data
print iris.target   # dependent (y) variables (if any) are stored as seperate arrays as .target
# .shape to get count of rows (n_samples) and columns (n_features) (150 rows/4 columns)
n_samples, n_features = iris.data.shape     

# Invoking models: In scikit-learn, an estimator for classification is a Python object that implements the methods fit(X, y) and predict(T).
from sklearn import svm
# for now treat the SVM like a black-box
clf = svm.SVC(gamma=0.001, C=100.)   

# when fitting we will only use about half the dataset. hence index data[:-899]       
clf.fit(iris.data[:-75], iris.target[:-75])     # clf.fit(X, y)
# to predict using the model use .predict:
predicted = clf.predict(iris.data)

# Compare predicted to actual:
actual = iris.target

#############################################################################################################
# 2. Classification: k-Nearest neighbours
############################################################################################################# 
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data          # converts the x variables to a numpy array
iris_y = iris.target        # converts the y variable to a numpy array 
len(iris_X)                 # note, there are 150 observations

# Split iris data in train and test data through random permutation
np.random.seed(0)
indices = np.random.permutation(len(iris_X))  # creates a random array of numbers
iris_X_train = iris_X[indices[:-75]]          # randomly pulls in half the dataset
iris_y_train = iris_y[indices[:-75]]
iris_X_test  = iris_X[indices[-75:]]
iris_y_test  = iris_y[indices[-75:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

predicted_train = knn.predict(iris_X_train)
actual_train = iris_y_train
accuracy_train = np.mean(predicted_train == actual_train)       # this is the training model accracy

# Get the Confusion Matrix
from sklearn import metrics
cm = DataFrame(metrics.confusion_matrix(predicted_train, actual_train),
                   index=[['Actual','Actual','Actual'],[0,1,2]], 
                   columns=[['Pred','Pred','Pred'],[0,1,2]])
                    
predicted_test = knn.predict(iris_X_test)
actual_test = iris_y_test
accuracy_test = np.mean(predicted_test == actual)               # this is the test model accuracy








