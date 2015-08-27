---
# Predicting wine ratings based on consumer ratings 
###By Jason Macwan###
####Thursday, Aug 20, 2015####
---

Python has an excellent library called scikit-learn that contains numerous algorithms that can be used in predictive analytics. Today, we are going to take a break from R and use Python instead to build a predictive model to forecast the rating of wine. The sample dataset was obtained from University of California Irvine's Machine Learning Repository. This dataset contains data on close to 7,000 wines, including the wine rating as well as other characteristics such as, sulphite content, acidity and alcohol level. For this article, we will only be looking at red wines and I have pre-processed the dataset to only include the reds. 

Let's begin by importing the dataset and having a look at it:

    import numpy as np
    import pandas as pd
    import sklearn
    
    wine = pd.read_csv("wine.csv", sep=',')
    # load the scikit-learn boston dataset (all sample datasets are 2d arrays)
    # Let's remove high_quality and color - not needed in this example
    wine = wine.drop(['high_quality','color'],1)

Let's take a look at the dataset and run some basic summary statistics. 

    # About the wine dataset
    wine.describe()

The output looks like this:

    Out[3]: 
       		fixed_acidity  volatile_acidity  citric_acid  residual_sugar  
    count	6497.000000   6497.000000  6497.000000 6497.000000   
    mean	7.215307  0.339666 0.3186335.443235   
    std 	1.296434  0.164636 0.1453184.757804   
    min 	3.800000  0.080000 0.0000000.600000   
    25% 	6.400000  0.230000 0.2500001.800000   
    50% 	7.000000  0.290000 0.3100003.000000   
    75% 	7.700000  0.400000 0.3900008.100000   
    max		15.900000  1.580000 1.660000 65.800000   
    
    		 chlorides  free_sulfur_dioxide  total_sulfur_dioxide  density  
    count  	6497.000000  6497.000000   6497.000000  6497.000000   
    mean 	 0.05603430.525319115.744574 0.994697   
    std  	 0.03503417.749400 56.521855 0.002999   
    min	  	 0.009000 1.000000  6.000000 0.987110   
    25% 	 0.03800017.000000 77.000000 0.992340   
    50% 	 0.04700029.000000118.000000 0.994890   
    75% 	 0.06500041.000000156.000000 0.996990   
    max 	 0.611000   289.000000440.000000 1.038980   
    
    		pHsulphates  alcohol  quality   is_red  
    count  6497.000000  6497.000000  6497.000000  6497.000000  6497.000000  
    mean  3.218501 0.53126810.491801 5.818378 0.246114  
    std   0.160787 0.148806 1.192712 0.873255 0.430779  
    min   2.720000 0.220000 8.000000 3.000000 0.000000  
    25%   3.110000 0.430000 9.500000 5.000000 0.000000  
    50%   3.210000 0.51000010.300000 6.000000 0.000000  
    75%   3.320000 0.60000011.300000 6.000000 0.000000  
    max   4.010000 2.00000014.900000 9.000000 1.000000  

At this point, let's do some basic data processing to before we build our predictive model.

    # seperate out the target variable and the predictors
    target = wine['quality']
    data = wine.drop('quality',axis=1)
    
    # Cross checks 
    print data.shape, target.shape
    print type(data), type(target)
    target.value_counts()
    data.describe()

We will also use k-fold cross-validation to split the data into test and training datasets. Train to build the model and test to test the model!

    # Use crossvalidation to create train and test datasets
    import sklearn.cross_validation
    xtrain, xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(data, target, train_size=0.8)
    
Now let's double check our test and train data:

     # Basic crosschecks
    print xtrain.shape, ytrain.shape
    print xtest.shape, ytest.shape

We see that we have 5,197 observations for the train dataset and 1,300 for the test dataset. We have in total 12 variables and we have also seperated our target variable (wine rating) into a separate Python object. 

    (5197, 12) (5197L,)
    (1300, 12) (1300L,)

At this point, let's build the model using Lasso Regression - a variant of linear regression.

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

The output looks something like this:

    RIDGE REGRESSION-------------------------------------------------------------
    The alpha (L1) level selected: 0.000464158883361
    Number of coefficients: 12
    
    Ridge: R-Square (training) = 0.292325961076
    Ridge: RMSE (training) = 0.729842956925
    Ridge: R-Square (test) = 0.310346258046
    Ridge: RMSE (test) = 0.743295460733
    
    Intercept: 93.8471405021
    Ridge: Coefficients: 
    	 Features 			 Coefficients
    0	fixed_acidity 		 0.086263
    1   volatile_acidity 	-1.431420
    2	citric_acid	  	    -0.042853
    3 	residual_sugar 		 0.057129
    4	chlorides 			-0.637858
    5	free_sulfur_dioxide  0.005205
    6   total_sulfur_dioxide -0.001507
    7	density				-93.338823
    8	pH 					 0.475868
    9	sulphates 			 0.707676
    10	alcohol  			 0.233484
    11	is_red  			 0.292575
    --------------------------------------------------------------------------------

### Conclusion
Ultimately, this model can help explain 30% of the variation in ratings (R-square=0.29). This is not very good for predictive purposes, but realistic nonetheless considering how small the dataset is. However, it's a good example of how regression techniques can be used in classifying wines and deciding which types to produce. For example, from the coefficients table we see that characteristics such as volatile acidity and density contribute negitively towards ratings while pH and sulphate content contribute positively towards ratings. With a larger dataset with more observations and variables, we could increase the R-square value and be able to take the coefficients table more seriously. 

For more predictive modelling examples, please visit:

[http://www.rpubs.com/macwanjason](http://www.rpubs.com/macwanjason)

