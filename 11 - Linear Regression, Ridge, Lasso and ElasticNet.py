import numpy as np
import pandas as pd
import sklearn

wine = pd.read_csv("wine.csv", sep=',')
# load the scikit-learn boston dataset (all sample datasets are 2d arrays)

# About the wine dataset
wine.describe()
# Let's remove high_quality and color - not needed in this example
wine = wine.drop(['high_quality','color'],1)

# list the number of NAs for each column (no NA's in the wine dataset)
for column_name in wine.columns:
    n_nans = wine[column_name].isnull().sum()
    if n_nans > 0:
        print column_name, n_nans

# seperate out the target variable and the predictors
target = wine['quality']
data = wine.drop('quality',axis=1)

# Cross checks 
print data.shape, target.shape
print type(data), type(target)
target.value_counts()
data.describe()

# Use crossvalidation to create train and test datasets
import sklearn.cross_validation
xtrain, xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(data, target, train_size=0.8)

# Basic crosschecks
print xtrain.shape, ytrain.shape
print xtest.shape, ytest.shape

#############################################################################################################
# 1. Ordinary Least Squares (OLS) Linear Regression 
#############################################################################################################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

linreg = LinearRegression()
linreg.fit(xtrain, ytrain)

# Note: roc_auc_score only works if y is a categorical variable
# Train dataset performance
lr_train_pred = linreg.predict(xtrain)
lr_train_perf = roc_auc_score(ytrain, lr_train_pred)
lr_train_error = mean_squared_error(ytrain, lr_train_pred)

# Test dataset performance
lr_test_pred = linreg.predict(xtest)
lr_test_perf = roc_auc_score(ytest, lr_test_pred)
lr_test_error = mean_squared_error(ytest, lr_test_pred)

print 'OLS: Area under the ROC curve (training) = {}'.format(lr_train_perf)
print 'OLS: The mean square error (training) = {}'.format(lr_train_error)
print 'OLS: Area under the ROC curve (test) = {}'.format(lr_test_perf)
print 'OLS: The mean square error (test) = {}'.format(lr_test_error)

#############################################################################################################
# 2. Ridge Regression
#############################################################################################################
# The parameter alpha increases L1 penalty when smaller. Alpha = 0 is linear regression. 
# RidgeCV incorporates iterating through many alphas and CV as well.
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

ridge = RidgeCV(alphas=np.logspace(-10, 10, 10),normalize=True, cv=10)   
ridge.fit(xtrain, ytrain)

# Train dataset performance
ridge_train_pred = ridge.predict(xtrain)
ridge_train_r2 = r2_score((ytrain), ridge_train_pred)
ridge_train_error = np.sqrt(mean_squared_error(ytrain, ridge_train_pred))

# Test dataset performance
ridge_test_pred = ridge.predict(xtest)
ridge_test_r2 = r2_score((ytest), ridge_test_pred)
ridge_test_error = np.sqrt(mean_squared_error(ytest, ridge_test_pred))

# Build coefficients table
from pandas import DataFrame
ridgecoeff = DataFrame(data.columns, columns = ['Features'])
ridgecoeff['Coefficients'] = ridge.coef_


print 'RIDGE REGRESSION    -------------------------------------------------------------'
print '\nThe alpha (L1) level selected: {}' .format(ridge.alpha_)
print 'Number of coefficients: {}' .format(len(ridge.coef_))
print '\nRidge: R-Square (training) = {}'.format(ridge_train_r2)
print 'Ridge: RMSE (training) = {}'.format(ridge_train_error)
print 'Ridge: R-Square (test) = {}'.format(ridge_test_r2)
print 'Ridge: RMSE (test) = {}'.format(ridge_test_error)
print '\nIntercept: {}' .format(ridge.intercept_)
print 'Ridge: Coefficients: \n {}' .format(ridgecoeff)
print '\n--------------------------------------------------------------------------------'


#############################################################################################################
# 2. Lasso
#############################################################################################################
# The parameter alpha increases L1 penalty when smaller. Alpha = 0 is linear regression. 
# LassoCV incorporates iterating through many alphas and CV as well. Also Lasso can be used for dimensionality 
# reduction as it is able to set coefficients to 0. 
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lasso = LassoCV(alphas=np.logspace(-10, 10, 10), normalize=True, cv=10, positive=False)  
lasso.fit(xtrain, ytrain)

# Train dataset performance
lasso_train_pred = lasso.predict(xtrain)
lasso_train_r2 = r2_score((ytrain), lasso_train_pred)
lasso_train_error = np.sqrt(mean_squared_error(ytrain, lasso_train_pred))

# Test dataset performance
lasso_test_pred = lasso.predict(xtest)
lasso_test_r2 = r2_score((ytest), lasso_test_pred)
lasso_test_error = np.sqrt(mean_squared_error(ytest, lasso_test_pred))

# Build coefficients table
from pandas import DataFrame
lassocoeff = DataFrame(data.columns, columns = ['Features'])
lassocoeff['Coefficients'] = lasso.coef_

print 'LASSO  ------------------------------------------------------------------------'
print '\nThe alpha (L1) level selected: {}' .format(lasso.alpha_)
print 'Number of coefficients: {}' .format(len(lasso.coef_))
print '\nLasso: R-Square (training) = {}'.format(lasso_train_r2)
print 'Lasso: RMSE (training) = {}'.format(lasso_train_error)
print '\nLasso: R-Square (test) = {}'.format(lasso_test_r2)
print 'Lasso: RMSE (test) = {}'.format(lasso_test_error)
print '\nIntercept: {}' .format(lasso.intercept_)
print 'Lasso: Coefficients: \n {}' .format(lassocoeff)
print '\n------------------------------------------------------------------------------'

#############################################################################################################
# 2. Elastic Net combines both L1 (Ridge) and L2 (Lasso) penalty estimators
############################################################################################################# 
# Like Lasso, Elastic Net can e used for dimensionality reduction

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

elastic = ElasticNetCV(alphas=np.logspace(-10, 10, 10), normalize=True, cv=10)  
elastic.fit(xtrain, ytrain)

# Train dataset performance
elastic_train_pred = elastic.predict(xtrain)
elastic_train_r2 = r2_score((ytrain), elastic_train_pred)
elastic_train_error = np.sqrt(mean_squared_error(ytrain, elastic_train_pred))

# Test dataset performance
elastic_test_pred = elastic.predict(xtest)
elastic_test_r2 = r2_score((ytest), elastic_test_pred)
elastic_test_error = np.sqrt(mean_squared_error(ytest, elastic_test_pred))

# Build coefficients table
from pandas import DataFrame
elasticcoeff = DataFrame(data.columns, columns = ['Features'])
elasticcoeff['Coefficients'] = elastic.coef_

print 'ELASTIC NET  -------------------------------------------------------------------'
print '\nThe alpha (L1) level selected: {}' .format(elastic.alpha_)
print 'Number of coefficients: {}' .format(len(elastic.coef_))
print '\nRidge: R-Square (training) = {}'.format(elastic_train_r2)
print 'Ridge: RMSE (training) = {}'.format(elastic_train_error)
print '\nRidge: R-Square (test) = {}'.format(elastic_test_r2)
print 'Ridge: RMSE (test) = {}'.format(elastic_test_error)
print '\nIntercept: {}' .format(elastic.intercept_)
print 'Ridge: Coefficients: \n {}' .format(elasticcoeff)
print '\n-------------------------------------------------------------------------------'




