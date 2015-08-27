# Load all required libraries
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import pylab as pl
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#############################################################################################################
# 1. Data Processing
#############################################################################################################

credit_transformed = pd.read_csv('D:\\Google Drive\\Documents\\Coding\\_Datasets\\Kaggle Credit Scoring\\credit_transformed.csv', 
                     sep=',')

# seperate out the target variable and the predictors
target = credit_transformed['Default']
data = credit_transformed.drop(['Default'],1) 

# Use stratified cross validation to create train and test datasets
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(np.array(target), n_iter=10, test_size=0.25)
      
for train_index, test_index in sss:
    xtrain, xtest = data.iloc[train_index], data.iloc[test_index]
    ytrain, ytest = target.iloc[train_index], target.iloc[test_index]
    
# Check target series
ytrain.value_counts()
ytest.value_counts()

#-----------------------------------------------------------------------------------------------------
'''NOTE ABOUT GRID SEARCH: Grid Search allows for automated optimization of a model's parameters while using Cross Validation when building the model. Once you do Grid Search, you will still have to rebuild the model with the optimized parameters found by Grid Search. This is because some of the functions associated with a model (e.g. Feature Importance for CART) are unavailable for the object created by Grid Search.  
'''
 
#############################################################################################################
# 1. Logistic Regression
############################################################################################################# 
# Scikit's Logistic Regression supports L1 and L2 regularization, multi-class target variables and class weighting!

logregGS = GridSearchCV(LogisticRegression(),
                    {'penalty':['l1','l2'], 'class_weight':['auto']})
logregGS.fit(xtrain,ytrain)
logregGS.best_params_  # obtain best parameters fron GridSearchCV

#Rerun model
logreg = LogisticRegression(penalty='l2', class_weight='auto')   
logreg.fit(xtrain,ytrain)

# Train dataset performance
logreg_train_pred = logreg.predict(xtrain)
logreg_train_perf = roc_auc_score(ytrain, logreg_train_pred)
logreg_train_accuracy = np.where(logreg_train_pred==ytrain, 1, 0).sum() / float(len(xtrain))
logreg_train_error = np.sqrt(mean_squared_error(ytrain, logreg_train_pred))
  
# Test dataset performance
logreg_test_pred = logreg.predict(xtest)
logreg_test_perf = roc_auc_score(ytest, logreg_test_pred)
logreg_test_accuracy = np.where(logreg_test_pred==ytest, 1, 0).sum() / float(len(xtest))
logreg_test_error = np.sqrt(mean_squared_error(ytest, logreg_test_pred))

cm_logreg = pd.DataFrame(confusion_matrix(ytest,logreg_test_pred),
                      index=[['Actual', 'Actual'],[0,1]],
                      columns = [['Predicted', 'Predicted'],[0,1]]) 
                      
# Build coefficients table
logreg_intercept = logreg.intercept_
logreg_coeffs = pd.DataFrame(logreg.coef_)
logreg_coeffs = logreg_coeff.transpose()
logregcoeff = DataFrame(data.columns, columns = ['Features'])
logregcoeff['Coefficients'] = logreg_coeffs[0]

                                        
# Accuracy may be overestimated if your target variable is not evenly distributed (seems to be the case here)
print '\nLOGISTIC REGRESSION------------------------------------------------------------'
print '\nLogistic Regression: Area under the ROC curve (training) = {}'.format(logreg_train_perf)
print 'Logistic Regression: Accuracy (training) = {}'.format(logreg_train_accuracy)
print 'Logistic Regression: RMSE (training) = {}'.format(logreg_train_error)
print '\nLogistic Regression: Area under the ROC curve (test) = {}'.format(logreg_train_perf)
print 'Logistic Regression: Accuracy (test) = {}'.format(logreg_train_accuracy)
print 'Logistic Regression: RMSE (test) = {}'.format(logreg_train_error)
print '\nConfusion Matrix (Test dataset)----------------------------'
print cm_logreg
print '\nIntercept: {}' .format(logreg.intercept_)
print 'Logistic Regression: Coefficients:\n {}' .format(logregcoeff)
print '\n-------------------------------------------------------------------------------'

#############################################################################################################
# 2. k Nearest Neighbours
############################################################################################################# 
# Alternative to GridSearchCV: Iterating to find the ideal k value
# As k goes up,the classifier is likely to be overfitting, or paying too much attention to the noise in the data. This is why we see accuracy decrease over increasing values of k. 
results = []
for n in range(1, 21, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(xtrain,ytrain)
    preds = clf.predict(xtest)
    accuracy = np.where(preds==ytest, 1, 0).sum() / float(len(xtest))
    print "k =  %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])
 
results = pd.DataFrame(results, columns=["n", "accuracy"])
 
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()
#------------------------------------------------------------------------------------------
# 1. Settle for k = 5
# Note: The kNN classifier requires all categorical variables be encoded as numbers.
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(xtrain,ytrain)

# Train dataset performance
knn_train_pred = knn.predict(xtrain)
knn_train_perf = roc_auc_score(ytrain, knn_train_pred)
knn_train_accuracy = np.where(knn_train_pred==ytrain, 1, 0).sum() / float(len(xtrain))
knn_train_error = np.sqrt(mean_squared_error(ytrain, knn_train_pred))

# Test dataset performance
knn_test_pred = knn.predict(xtest)
knn_test_perf = roc_auc_score(ytest, knn_test_pred)
knn_test_accuracy = np.where(knn_test_pred==ytest, 1, 0).sum() / float(len(xtest))
knn_test_error = np.sqrt(mean_squared_error(ytest, knn_test_pred))

cm_knn = pd.DataFrame(confusion_matrix(ytest,knn_test_pred),
                      index=[['Actual', 'Actual'],[0,1]],
                      columns = [['Predicted', 'Predicted'],[0,1]]);cm_knn  
                      
# Accuracy may be overestimated if your target variable is not evenly distributed (seems to be the case here)
print '\nk Nearest Neighbours-----------------------------------------------------------'
print '\nkNN: Area under the ROC curve (training) = {}'.format(knn_train_perf)
print 'kNN: Accuracy (training) = {}'.format(knn_train_accuracy)
print 'kNN: RMSE (training) = {}'.format(knn_train_error)
print '\nkNN: Area under the ROC curve (test) = {}'.format(knn_test_perf)
print 'kNN: Accuracy (test) = {}'.format(knn_test_accuracy)
print 'kNN: RMSE (test) = {}'.format(knn_test_error)
print '\nConfusion Matrix (Test dataset)----------------------------'
print cm_knn
print '\n-------------------------------------------------------------------------------'

#############################################################################################################
# 3. CART
############################################################################################################# 

cartGS = GridSearchCV(DecisionTreeClassifier(),
                    {'criterion':['gini','entropy'], 'max_depth':[None], 'class_weight':['auto']})
cartGS.fit(xtrain,ytrain)
cartGS.best_params_  # obtain best parameters fron GridSearchCV

#Rerun model
cart = DecisionTreeClassifier(criterion='gini')   
cart.fit(xtrain,ytrain)

# Train dataset performance
cart_train_pred = cart.predict(xtrain)
cart_train_perf = roc_auc_score(ytrain, cart_train_pred)
cart_train_accuracy = np.where(cart_train_pred==ytrain, 1, 0).sum() / float(len(xtrain))
cart_train_error = np.sqrt(mean_squared_error(ytrain, cart_train_pred))
  
# Test dataset performance
cart_test_pred = cart.predict(xtest)
cart_test_perf = roc_auc_score(ytest, cart_test_pred)
cart_test_accuracy = np.where(cart_test_pred==ytest, 1, 0).sum() / float(len(xtest))
cart_test_error = np.sqrt(mean_squared_error(ytest, cart_test_pred))

cm_cart = pd.DataFrame(confusion_matrix(ytest,cart_test_pred),
                      index=[['Actual', 'Actual'],[0,1]],
                      columns = [['Predicted', 'Predicted'],[0,1]]) 
                      
# Build feature importance
cartimp = DataFrame(data.columns, columns = ['Features'])
cartimp['Importance'] = cart.feature_importances_
cartimp.sort('Importance', ascending = False, inplace = True)
                     
# Accuracy may be overestimated if your target variable is not evenly distributed (seems to be the case here)
print '\nCART---------------------------------------------------------------------------'
print '\nCART: Area under the ROC curve (training) = {}'.format(cart_train_perf)
print 'CART: Accuracy (training) = {}'.format(cart_train_accuracy)
print 'CART: RMSE (training) = {}'.format(cart_train_error)
print '\nCART: Area under the ROC curve (test) = {}'.format(cart_train_perf)
print 'CART: Accuracy (test) = {}'.format(cart_train_accuracy)
print 'CART: RMSE (test) = {}'.format(cart_train_error)
print '\nConfusion Matrix (Test dataset)----------------------------'
print cm_cart
print '\nCART Variable Importance---------\n {}'.format(cartimp)
print '\n-------------------------------------------------------------------------------'

#############################################################################################################
# 4. Random Forests
############################################################################################################# 

rfGS = GridSearchCV(RandomForestClassifier(),
                    {'criterion':['gini','entropy'], 'max_depth':[None], 'class_weight':['auto']})
rfGS.fit(xtrain,ytrain)
rfGS.best_params_  # obtain best parameters fron GridSearchCV

# Rerun model
rf = RandomForestClassifier(criterion='entropy')   
rf.fit(xtrain,ytrain)

# Train dataset performance
rf_train_pred = rf.predict(xtrain)
rf_train_perf = roc_auc_score(ytrain, rf_train_pred)
rf_train_accuracy = np.where(rf_train_pred==ytrain, 1, 0).sum() / float(len(xtrain))
rf_train_error = np.sqrt(mean_squared_error(ytrain, rf_train_pred))
  
# Test dataset performance
rf_test_pred = rf.predict(xtest)
rf_test_perf = roc_auc_score(ytest, rf_test_pred)
rf_test_accuracy = np.where(rf_test_pred==ytest, 1, 0).sum() / float(len(xtest))
rf_test_error = np.sqrt(mean_squared_error(ytest, rf_test_pred))

cm_rf = pd.DataFrame(confusion_matrix(ytest,rf_test_pred),
                      index=[['Actual', 'Actual'],[0,1]],
                      columns = [['Predicted', 'Predicted'],[0,1]]) 
                      
# Build feature importance
rfimp = DataFrame(data.columns, columns = ['Features'])
rfimp['Importance'] = rf.feature_importances_
rfimp.sort('Importance', ascending = False, inplace = True)

print '\nRandom forest Classifier-------------------------------------------------------'
print '\nRandom Forest: Area under the ROC curve (training) = {}'.format(rf_train_perf)
print 'Random Forest: Accuracy (training) = {}'.format(rf_train_accuracy)
print 'Random Forest: RMSE (training) = {}'.format(rf_train_error)
print '\nRandom Forest: Area under the ROC curve (test) = {}'.format(rf_train_perf)
print 'Random Forest: Accuracy (test) = {}'.format(rf_train_accuracy)
print 'Random Forest: RMSE (test) = {}'.format(rf_train_error)
print '\nConfusion Matrix (Test dataset)---------------------'
print cm_rf
print '\nRandom Forests Variable Importance---------\n {}' .format(rfimp)
print '\n-------------------------------------------------------------------------------'

#############################################################################################################
# 5.Gradient Boosting Machine (GMB)
############################################################################################################# 

gbmGS = GridSearchCV(GradientBoostingClassifier(),
                    {'loss':['deviance','exponential'], 'n_estimators':[100], 
                    'max_depth':[3], 'subsample':[1]})
                
gbmGS.fit(xtrain,ytrain)
gbmGS.best_params_  # obtain best parameters fron GridSearchCV

# Rerun model
gbm = GradientBoostingClassifier(loss='exponential',max_depth=3,n_estimators=100,subsample=1)
gbm.fit(xtrain, ytrain)

# Train datazet performance
gbm_train_pred = gbm.predict(xtrain) 
gbm_train_perf = roc_auc_score(ytrain, gbm_train_pred) 
gbm_train_accuracy = np.where(gbm_train_pred==ytrain, 1, 0).sum() / float(len(xtest))
gbm_train_error = mean_squared_error(ytrain, gbm_train_pred)

# Train datazet performance
gbm_test_pred = gbm.predict(xtest)
gbm_test_perf = roc_auc_score(ytest, gbm_test_pred)
gbm_test_accuracy = np.where(gbm_test_pred==ytest, 1, 0).sum() / float(len(xtest))
gbm_test_error = mean_squared_error(ytest, gbm_test_pred)
    
cm_gbm = pd.DataFrame(confusion_matrix(ytest,gbm_test_pred),
                      index=[['Actual', 'Actual'],[0,1]],
                      columns = [['Predicted', 'Predicted'],[0,1]]) 
                      
# Build feature importance
gbmimp = DataFrame(data.columns, columns = ['Features'])
gbmimp['Importance'] = gbm.feature_importances_
gbmimp.sort('Importance', ascending = False, inplace = True)

# Accuracy may be overestimated if your target variable is not evenly distributed (seems to be the case here)
print '\nGradient Boosting Machine-------------------------------------------------------'
print '\nGBM: Area under the ROC curve (training) = {}'.format(gbm_train_perf)
print 'GBM: Accuracy (training) = {}'.format(gbm_train_accuracy)
print 'GBM: RMSE (training) = {}'.format(gbm_train_error)
print '\nGBM: Area under the ROC curve (test) = {}'.format(gbm_train_perf)
print 'GBM: Accuracy (test) = {}'.format(gbm_train_accuracy)
print 'GBM: RMSE (test) = {}'.format(gbm_train_error)
print '\nConfusion Matrix (Test dataset)----------------------------'
print cm_gbm
print '\nGBM Variable Importance---------\n {}' .format(gbmimp)
print '\n-------------------------------------------------------------------------------'



























