import numpy as np
import pandas as pd

credit = pd.read_csv('D:\\Google Drive\\Documents\\Coding\\_Datasets\\Kaggle Credit Scoring\\cs-training.csv', 
                     index_col=False, sep=',',skiprows=[0],header=None)
credit = credit.rename(columns={0:'Default',1:'UtilizationRate',2:'Age',
                             3:'Delay30to59',4:'DebtRatio',5:'Income',
                             6:'CredLines',7:'Delay90',8:'ReLines',
                             9:'Delay60to89',10:'Dependents'})
                             
#############################################################################################################
# 1. Basic Exploratory Analysis
#############################################################################################################
# About the credit dataset
print credit.describe()
credit.groupby('Income').count()  # groupby counts
credit['UtilizationRate'].describe()  # these are percentages; looking for outliers
credit['DebtRatio'].describe()   # these are percentages; looking for outliers
credit['Delay30to59'].value_counts()
credit['Delay60to89'].value_counts()
credit['Delay90'].value_counts()
credit['CredLines'].value_counts()
credit['Income'].describe()

# SORTING:
credit.sort(columns = ['UtilizationRate'],
        ascending = True, inplace = True) 

# CORRELATION
credittemp = credit.dropna()   # .corrcoef() won't work with NAs    
np.corrcoef(credittemp['Income'], credittemp['Dependents'])

#############################################################################################################
# 1. Outliers
#############################################################################################################
# UtilizationRate is a % so values (high is usually bad for your credit score)
credit['UtilizationRate'][credit['UtilizationRate']>100].describe()
# Ditto for DebtRatio (U.S average DR is 163%)
credit['DebtRatio'][credit['DebtRatio']>200].describe()

#############################################################################################################
# 2. Explore Missing Values
#############################################################################################################
# Look for NAs:
for column_name in credit.columns:
    n_nans = credit[column_name].isnull().sum()
    if n_nans > 0:
        print column_name, n_nans
            
# Shows rows that have NAs
credit[pd.isnull(credit).any(axis=1)]

# Create 'NoIncome' and 'NoDependents' : We need to run this because we will be setting all NAs fo 'income' and 'dependents' to 0, which also represents zero income and 0 dependents. This way we extract maximum information from the dataset.
credit['NoIncome'] = np.where(credit['Income'] == 0, 1, 0)
credit['NoDependents'] = np.where(credit['Dependents'] == 0, 1, 0)

# 4. Flag na 'income' and na 'dependents'
credit['UnknownIncome'] = 1
credit['UnknownDependents'] = 1
credit['UnknownIncome'][pd.notnull(credit['Income'])] = 0
credit['UnknownDependents'][pd.notnull(credit['Dependents'])] = 0

# Set all NA 'income' and 'dependents' to 0 because scikit models can't handle NAs
credit['Income'].fillna(0, inplace= True) 
credit['Dependents'].fillna(0, inplace= True)   

#############################################################################################################
# 5. Imputing variables
#############################################################################################################
# Binary Flag variables
# Utilization Rate----------------------------------------------------------

credit['FullUtilization'] = np.where(credit['UtilizationRate'] == 100, 1, 0)

credit['ExcessUtilization'] = np.where(credit['UtilizationRate'] > 100, 1, 0)

credit['ZeroUtilization'] = np.where(credit['UtilizationRate']== 0, 1, 0)

credit['LowUtilizationRate'] = np.where(credit['UtilizationRate'] < 4, 1, 0)

'''credit['MedUtilizationRate'] = 0
credit['MedUtilizationRate'][credit['UtilizationRate'] where range(4,11)] = 1'''

credit['HighUtilizationRate'] = np.where(credit['UtilizationRate'] < 10, 1, 0)

# Payment Past Due Date------------------------------------------------------
credit['TimesPastDue'] = credit['Delay30to59'] + credit['Delay60to89'] + credit['Delay90']

credit['NeverLate'] = np.where(credit['TimesPastDue'] == 0, 1, 0)

credit['HighDelay30to59'] = np.where(credit['Delay30to59'] > 90, 1, 0)

credit['HighDelay60to89'] = np.where(credit['Delay60to89'] > 90, 1, 0)

credit['HighDelay90'] = np.where(credit['Delay90'] > 90, 1, 0)

# Credit Lines----------------------------------------------------------------
credit['highCredLines'] = np.where(credit['CredLines'] >= 30, 1, 0)

# Debt Ratio
credit['LowDebtRatio'] = np.where(credit['DebtRatio'] < 60, 1, 0)

'''credit['MedDebtRatio'] = 0
credit['MedUtilizationRate'][credit['DebtRatio'] >= 60] & [credit['DebtRatio'] < 200 ]  = 1'''

credit['HighDebtRatio'] = np.where(credit['DebtRatio'] >= 200, 1, 0)

# Multiple Real Estate Lines/Loans
credit['MultipleReLines'] = np.where(credit['ReLines'] > 1, 1, 0)


# New imputed variables--------------------------------------------------------
LogIncome = np.log10(credit['Income'])
LogAge = np.log10(credit['Age'])
credit['TotalLines'] = credit['CredLines'] + credit['ReLines']
credit['MonthlyDebtPmt'] = credit['Income'] * credit['DebtRatio']
credit['TotalLines'] = credit['CredLines'] + credit['ReLines'] 
credit['Income*Age'] = credit['Income'] * credit['Age'] 

#############################################################################################################
# XX. Export dataset to csv
#############################################################################################################
credit.to_csv('D:\\Google Drive\\Documents\\Coding\\_Datasets\\Kaggle Credit Scoring\\credit_transformed.csv', sep=',', index=False)
 