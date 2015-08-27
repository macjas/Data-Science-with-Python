import numpy as np
my_list1 = [1,2,3,4]    
my_list2 = [11,22,33,44] 
my_lists = [my_list1, my_list2]  
  
#############################################################################################################
# 1. Introducing Numpy Arrays
#############################################################################################################           
# Arrays in Python/Numpy are Python's version of vectorized objects much like
# R's vectors. Numpy's array function can convert multiple lists into arrays. 
# Each list is converted into a row in the array.
arr = np.array(my_lists)              # create array from lists with the .array method        
print arr
arr.shape                             # the .shape method shows the dimensions of the array
arr.shape[0]                          # counts rows of 2D array
arr.shape[1]                          # counts columns of 2D array 
arr.dtype                             # shows the type of objects in the array
x = np.zeros((5,5))                   # the .zeros method creates an array with zeros (floating point)  
x = np.ones((5,5))                    # the .ones method creates an array/[matrix] with ones (floating point)   
np.empty(5)                           # the .empty method creates an empty array
np.eye(5)                             # .eye creates an identity matrix (5x5)
np.arange(5,50,2)                     # create array with range 5:50, with stepsize=2

# Arrays are vectorized for math operations
arr1 = np.array([[1,2,3],[8,9,10]])
print arr1 * arr1
print arr1 * 2

#############################################################################################################
# 2. Indexing Numpy Arrays
############################################################################################################# 
arr = np.arange(0,11)   
# 2D ARRAYS: Indexing an array is pretty much like indexing a list
print arr
print arr[8]                          # indexes the 8th element
print arr[0:5]                        # indexes elements 0,1,2,3,4
arr[0:5] = 100                        # like with lists, you can reassign arrays with indexing

# To optimize memory, Python does not create new arrays when slicing. To do so you have to explicitly use the .copy method
arr = np.arange(0,11)
slice_of_array = arr[0:6]
slice_of_array[:] = 99
print arr                             # notice, slice_of_array was just a view of arr!!

# To slice and create a new array use the .copy method!
arr = np.arange(0,11)
slice_of_array = arr[0:6]
slice_of_array = slice_of_array.copy()  # this makes 'slice_of_array' a new array instead of just a view!
slice_of_array[:] = 99
print arr 
print slice_of_array

# 3D ARRAYS:
arr2d = np.arange(50).reshape((10,5))  #.reshape coverts a 2D array into a 3D array
arr2d

arr2d[3]                               # indexing rows
arr2d[0][4]                            # indexing an element [row][column]
arr2d[:3,2:]                           # indexing a block
arr2d[[6,2,1,9]]                       # fancy indexing: index any row in any order!

#############################################################################################################
# 3. Processing Numpy Arrays
#############################################################################################################
arr = np.arange(50).reshape((10,5))
arr1 = arr.T                           # transposes arays
arr2 = arr1.swapaxes(0,1)              # swaps rows and columns

# Universal functions on arrays:
#For full and extensive list of all universal functions
website = "http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs"
import webbrowser
webbrowser.open(website)

print np.sqrt(arr)                     # square root of entire array
print np.exp(arr)                      # e^ of entire array
a = np.random.randn(10)                # creates an array with random nos from a gaussian distribution
b = np.random.randn(10) 
print a.sum()                          # sums the entire array
print a.mean()                         # mean of all elements in the array
print a.std()                          # sd of all elements in the array
print a.var()                          # variance of all elements in the array
print a.sort()                         # sorts an array
print np.add(a,b)                      # adds two arrays
print np.maximum(a,b)                  # outputs max of two arrays
print np.minimum(a,b)                  # outputs max of two arrays

# Boolean arrays:
bool_arr = np.array([True,False,True])
bool_arr.any()                         # returns True if atlease one element is True
bool_arr.all()                         # returns True if all elements are True

# Conditional Statement method:
A = np.array([1,2,3,4])
B= np.array([100,200,300,400])
condition = np.array([True,True,False,True])
print np.where(condition,A,B)         # outputs elements from A if True and B if False

countries = np.array(['France', 'Germany', 'USA', 'Russia','USA','Mexico','Germany'])
print np.unique(countries)            # returns unique elements in an array
np.in1d(['Sweden'],countries)         # to check if an element exists in an array

 