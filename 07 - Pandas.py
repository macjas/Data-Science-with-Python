import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randn

#############################################################################################################
# 1. Series Basics
#############################################################################################################
# Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index.

s = Series([3,6,9,12])                             # use the Series function to create series
s.values                                           # returns the values in the series
s.index                                            # returns the index of the series

gdp = Series([8700000,4300000,3000000],            # you can specify names/strings for indexes
              index= ['USSR','Germany','China'])
       
gdp['USSR'] == gdp[0]                              # indexing works with either the index labels or the index number
gdp[gdp > 5000000]                                 # you can use inequality operators to check/filter 
'USA' in gdp                                       # to check if a specific index exists
gdp_dict = gdp.to_dict()                           # .to_dict() converts a series to a dictionary
gdp2 = Series(gdp_dict)                            # and dictionaries can be converted to series

gdp2 = Series(gdp, 
       index=['USSR','Germany','China','USA'])               
pd.isnull(gdp2)                                    # .isnull() is used to find NaNs and Nulls
pd.notnull(gdp2)                                   # opposite of .isnull()
gdp3 = gdp + gdp2                                  # series are vectorized: you can +,-,*,/ etc.
gdp3.index.name = 'Country'                        # you can give the index a header
gdp3 = gdp3.drop('USA')                            # delete a row with .drop()

# .forwardfill (ffill) interpolates values between indices from top-down
ser3 = Series(['USA','Mexico','Canada'],index=[0,5,10])
ser4 = ser3.reindex(range(15),method='ffill')
print ser3, ser4

# .reindex() to rearrange data to a new index
ser1 = Series([1,2,3,4], index = ['A', 'B', 'C', 'D'])
ser2 = ser1.reindex(['A','B','C','D','E','F']) 
print ser1, ser2 

#############################################################################################################
# 2. Dataframes Basics
#############################################################################################################
# The dataframe is our workhorse. It is a two-dimensional labeled array capable of holding any data type.

df_columns = ['col1','col2','col3','col4','col5'] 
new_columns = ['col1','col2','col3','col4','col5','newCol'] 
a_list = [1,2,3,4,5,6]
df = DataFrame(randn(25).reshape((5,5)),
               index = ['E','F','D','A','B'], 
               columns = df_columns)
print df

# Indexing
df.dtypes                                            # check the data format in the dataframe
df.col3.dtype                                        # check the data format for a specific column 
df.columns.values                                    # returns a numpy array with a list of all columns                 
df[['col2','col1']]                                  # index a column by it's name                                                 
df[0:1]                                              # to index a row use ranges
df.ix[0]                                             # .ix() indexes rows but outputs the row as a column
df.ix[1:2,] ['col1']] = 666                          # to index or modify a specific cell
df3 = df.ix[['A','B','C','D','E','F'],new_columns]   # .ix() can be used to add rows and columns simultaneously
df2 = df.reindex(['A','B','C','D','E','F'])          # .reindex() to add new rows (in this case 'C')
df2 = df.reindex(columns=new_columns)                # .reindex() can also add new columns
df3['newCol'] = a_list                               # you can use lists to fill columns
df4 = df3.drop('newCol', axis = 1)                   # drop a column 
df5 = df4.rename(columns={'col1':'test1'})           # rename a column 
df6 = df5.rename(index={'A':'Alpha'})                # rename an index
dflist = df['col2'].tolist()                         # convert a column to a list
df6.index = df6.index.map(str.lower)                 # bulk convert all indexes to lower case
df7 = df6.rename(index=str.title, 
                 columns=str.title)                  # bulk convert indexes and columns to capital first letter
df > 0                                               # conditional operators create new, boolean dataframes
df.isnull()                                          # check entire dataframe for NaNs


#############################################################################################################
# 3. Sorting Dataframes
#############################################################################################################
df.sort_index()                                      # sort descending by index
df.sort(columns = ['Col1','Col2'],
        ascending = True, inplace = True)            # sort by specified column

#############################################################################################################
# 4. Missing Values
#############################################################################################################

df5 = DataFrame([[1,2,3,np.nan],[np.nan,5,6,7],[7,np.nan,9,np.nan],[np.nan,np.nan,np.nan,np.nan]])

df5.isnull()                                         # outputs a boolean dataframe
df6 = df5.dropna()                                   # .dropna() default drops a row if it finds even one NA
df7 = df5.dropna(how = 'all')                        # .dropna() will only dorp row if all obs are NA
df8 = df5.dropna(axis = 1, thresh = 2)               # to scan across columns; thresh = min count of NA's before deleting
df9 = df5.fillna(value = 0, axis = 1)                # .fillna() scans columns for NAs, replaces them with 0           
df10 = df5.fillna({0:0,1:1,2:2,3:3})                 # use a dictionary to fill diff values for diff columns
df5.fillna('NA', inplace= True )                     # use inplace=True for permanent processing


#############################################################################################################
# 5. Index Hierarchies: pandas allows you to create multiple levels of indexes
#############################################################################################################
city = DataFrame(np.arange(16).reshape(4,4),
                    index=[['a','a','b','b'],[1,2,1,2]],
                    columns=[['NY','NY','LA','SF'],['cold','hot','hot','cold']])

city.index.names = ['Level-0','Level-1']            # name the indexes
city.columns.names = ['Cities','Temp']              # name the columns
city2 = city.swaplevel('Cities','Temp',axis=1)      # swap level for columns
city.sortlevel(1)                                   # sort by Level-1
city3 = city.sum(level='Temp', axis=1)              # math operations by levels - sums the rows
city4 = city.sum(level='Level-0', axis=0)           # math operations by levels - sums on columns
                                                 
#############################################################################################################
# 6. Importing Data
#############################################################################################################
# Getting data intp Python like SAS's CARDS system
                                                 
from StringIO import StringIO
data ="""\
Sample Type Intelligence
1 Dog Smart
2 Dog Smart
3 Cat Dumb
4 Cat Dumb
5 Dog Dumb
6 Cat Smart"""

animals = pd.read_table(StringIO(data),sep='\s+')   #Store as dframe

# Importing text or csv files is done with either .read_table() or .read_csv(). Use '\\' when specifying directories to avoid confusion with escape sequences
credit = pd.read_csv("D:\\Google Drive\\Documents\\Coding\\Python\\Data\\germancredit.csv",
                     sep=',') 
credit.head()                                       # lists the first 5 records
credit.tail()                                       # lists the last 5 records   
credit.describe()                                   # summary statistics!                                

# JSON:
json_obj = """
{   "zoo_animal": "Lion",
    "food": ["Meat", "Veggies", "Honey"],
    "fur": "Golden",
    "clothes": null, 
    "diet": [{"zoo_animal": "Gazelle", "food":"grass", "fur": "Brown"}]
}
"""
import json                                         # Let import json module
data = json.loads(json_obj)                         # Lets load json data
json.dumps(data)                                    # we can also convert back to JSON
dframe = DataFrame(data['diet'])                    # loading JSON data into a dataframe

# HTML Scraping
url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'
banks = pd.io.html.read_html(url)
banks[0]

#############################################################################################################
# 7. Data Joins
#############################################################################################################
employee = DataFrame({'ID':['100','101','102','108'],'Name':['Jason','John','Josh','Lucas']})
sales = DataFrame({'ID':['100','102','101','102','101','101', '103'],'Sales': range(50,57)})
bonus = DataFrame({'ID':['100','100','100','102','101','108','105'],
                   'Bonus': [1000, 400, 500, 2000, 4300, 200, 50]})


mergeL = pd.merge(employee, sales, on = 'ID', how = 'left')       # A Left inner Join
mergeR = pd.merge(employee, sales, on = 'ID', how = 'right')      # A Right inner Join
mergeO = pd.merge(employee, sales, on = 'ID', how = 'outer')      # An Outer Join
mergeM = pd.merge(sales, bonus, on = 'ID')                        # A many-to-many Join
stack = pd.concat([employee, sales], ignore_index = True)         # Vertical Stacking

#############################################################################################################
# 8. Reshaping & Pivoting
#############################################################################################################
df1 = DataFrame([['Big','LAX',3,np.nan],['Big','SFO',6,7],['Med','SEA-TAC',9,np.nan],['Small','POR',np.nan,np.nan]],
                 index=pd.Index(['LA', 'SF', 'SEA', 'POR']),
                 columns=pd.Index(['Type', 'Airport', 'Cool Factor','D']))

# .unstack(): used to convert columns into rows and into a hierarchical index 
df2 = df1.stack(dropna = False)                    # converts columns into the child index
df3 = df1.unstack()                                # converts columns into the parent index 

# .pivot(index, columns, values) is used to reshape data like dplyr in R
df4 = df1.pivot('Airport','Type','Cool Factor')    # yes! its that easy to reshape!

#############################################################################################################
# 9. Outlier Analysis
#############################################################################################################
np.random.seed(12345)
df = DataFrame(np.random.randn(1000,4))
df.describe()                                        # assume outliers are in the -+3 region

df[0][np.abs(df[0])>3]                               # show all rows in column 0 that are > abs(3)
df[(np.abs(df)>3).any(1)]                            # show all values in the dataframe that are > abs(3)
df[np.abs(df)>3] = np.sign(df) * 3                   # caps all values > abs(3) to 3; .sign()                                

#############################################################################################################
# 10. Binning Data
#############################################################################################################
years = [1990,1991,1992,2008,2012,2015,
         1987,1969,2013,2008,1999]
bins = [1960,1970,1980,1990,2000,2010,2020]

### .cut() bins the data in 'years' into a Panda object called Categories
### bins: is a list the specifies the end points of the class intervals
### right: argument specifies if the right edge in inclusive or not'''                    

# puts the 'years' data into class intervals of decades
cat1 = pd.cut(years,bins, right = False)
# puts the 'years' data into 4 equal width class intervals 
cat3 = pd.cut(years, 4, right = False)  
cat1.categories                                       # lists the categories in the variable cat
pd.value_counts(cat1)                                 # .value_counts() counts the observation in each class interval

#############################################################################################################
# 11. Data Aggregation
#############################################################################################################
# Data Sets
from StringIO import StringIO
data ="""\
Sample Type Intelligence
1 Dog Smart
2 Dog Smart
3 Cat Dumb
4 Cat Dumb
5 Dog Dumb
6 Cat Smart"""
animals = pd.read_table(StringIO(data),sep='\s+')

wine = pd.read_csv('wine.csv', sep=',')

df = DataFrame({'k1':['X','X','Y','Y','Z'],
                'k2':['alpha','beta','alpha','beta','alpha'],
                'dataset1':np.random.randn(5),
                'dataset2':np.random.randn(5)})

df2 = DataFrame(np.arange(16).reshape(4, 4),
                   columns=['W', 'X', 'Y', 'Z'],
                   index=['Dog', 'Cat', 'Bird', 'Mouse'])
df2.ix[1:2, ['W', 'Y']] = np.nan 

#-----------------------------------------------------------------------
# .groupby()
                
df.groupby('k1').mean()                        # group all number columns by 'k1' and show mean; returns a series
df['dataset1'].groupby(df['k1']).mean()        # only show aggregate 'dataset1' column
df.groupby(['k1','k2']).mean()                 # with two group by's we get a hierarchical index dataframe 
wine2 = wine.groupby('quality').describe()     # groupby 'quality' and show summary statistics; like PROC UNIVARIATE
wine3 = wine.groupby('quality').count()        # groupby 'quality' and show counts; like PROC FREQ


# You can also groupy by a new set of factors by creating a dictionary mapping an existing column to a the new factors
behavior_map = {'W': 'good', 'X': 'bad', 'Y': 'good','Z': 'bad'}
df3 = df2.groupby(behavior_map, axis=1).sum()        # this a groupby and sum function

# You can create your own functions for groupby:

# Create a groupby Ranker function (like PROC RANK):
def ranker(df):
    df['qualityRank'] = np.arange(len(df)) + 1
    return df
    
# Now sort the dframe by alcohol in ascending order
wine.sort('alcohol', ascending=False, inplace=True)

# Now we'll group by quality and apply our ranking function
wine4 = wine.groupby('quality').apply(ranker) 
# Summary statistics by qualityRank
wine4[wine4['qualityRank'] == 1].describe()

# .crosstab()

pd.crosstab(animals.Type, animals.Intelligence, margins=True)


#############################################################################################################
# 12. Data Imputation
#############################################################################################################

# Derived fields based on indexing
wine['qual/alc ratio'] = wine['quality']/wine['alcohol']

# Derived fields based on conditional logic 
def impute(row):
    if row['quality'] > 1 and row['quality'] <= 4:
        return 'Poor'
    elif row['quality'] > 4 and row['quality'] <= 6:
        return 'Average'
    else:
        return 'Good'
wine['Review'] = wine.apply(impute,axis=1)               # apply function to the dataframe


rdd = sc.parallelize([[1, 1, 2, 3],[1,8,9,10],[4,5,6,7]])
test = rdd.distinct()

[1,8,9,10]
[4,5,6,7]





















