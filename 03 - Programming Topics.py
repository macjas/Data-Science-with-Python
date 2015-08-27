#############################################################################################################
# 1. Conditional Statements
#############################################################################################################
# Unlike other languages, Python uses indentation as part of its code structure. Below, the two print statements are called a block of code. The block ends when Python encounters a line of code which is indented one level up.

x = 0.5
if x > 0:
    print "This is the first block"
    print "x is positive"
print "This isn't part of the block and will always print"

x = 2
if x > 0:
    print "This is the first block"
    print "x is positive"
print "This isn't part of the block and will always print"

# You can have as many elif clauses as you want, but at most one else clause, and the else clause must always come last. In addition, Python will only execute the first block of code that matches. In the code below the second print statement will not print!

x = 1.5
if x > 1:
    print "x is greater than 1"
elif x >= 0:                      # in this case you could have also used 'else'
    print "x is non-negative"

'''Use two separate 'if' statements if you need two matches or nested statements'''
if x > 1:
    print "x is greater than 1"
if x >= 0:
    print "x is non-negative"

if x >= 0:
    if x > 1:
        print "x is greater than 1"
    print "x is non-negative"

#############################################################################################################
# 2. Looping
#############################################################################################################
# WHILE LOOPS:  In contrast to an if statement where the block executes either 0 or 1 times depending on whether the test expression is True or False, a while loop executes indefinitely as long as the test expression is True. 

i = 0
total = 0
while i < 5:
    total = total + i
    i = i + 1
print total

# You can loop over sets, lists and even strings as well. This repeatedly pops an element from the set of plays until the set of plays is empty.

plays = set(['Hamlet', 'Macbeth', 'King Lear'])
while plays:
    play = plays.pop()
    print 'Perform', play
    
name = "Jason Macwan"
for i in name:
    print i

# FOR LOOPS: The for loop is designed to loop over sequences, like lists, sets, dictionaries, and even strings. Notice that the loop does not interfere with the values in the list. In fact this is a very important restriction on for loops: a for loop should not try to modify the sequence it is looping over, or you may get unexpected results! A very common pattern is to use the range() function to create a list of numbers to loop over.

# Below, the variable i will be assigned the value of each list item (in this case the range) in turn.
total = 0
for i in xrange(100000):          # The xrange function is a more efficient function used with for loops
    total = total + i
print total

for i in range(1,7):              # remember range(1,7) = 1,2,3,4,5,6
    print i 
    
# FOR LOOPS OVER DICTIONARIES:
webster = {
	"Aardvark" : "A star of a popular children's cartoon show.",
    "Baa" : "The sound a goat makes.",
    "Carpet": "Goes on the floor.",
    "Dab": "A small amount."}
# A dictionary is unordered, so there is no guaranteed sequence in which the items will be accessed if you loop through the keys. Therefore, sort the dictionary before looping through it
webster_keys = webster.keys()
webster_keys.sort()
for i in webster_keys:
    print webster.get(i)         # this will print the values for each k-v pair
    print i                      # this will print the keys for each k-v pair
   
# FOR LOOPS OVER LISTS: Method 1 is useful to loop through the list, but it's not possible to modify the list this way. Method 2 uses indexes to loop through the list, making it possible to also modify the list if needed.
list_a = ['q','w','e','r','t']  

for i in list_a:                 # method 1
    print i
    
for i in xrange(0, len(list_a)):  # method 2
    print list_a[i] + '**'

# It's also common to need to iterate over two lists at once. This is where the built-in zip function comes in handy. Zip will create pairs of elements when passed two lists, and will stop at the end of the shorter list.
list_a = [3, 9, 17, 15, 19]
list_b = [2, 4, 8, 10, 30, 40, 50, 60, 70, 80, 90]
        
for a, b in zip(list_a, list_b):
    if a > b:
        print a
    else:
        print b

# FOR IF CONTINUE: If execution hits a continue statement, then the execution will jump immediately to the start of the next iteration of the loop. This is useful if you want to skip occasional values

values = [7, 6, 4, 7, 19, 2, 1]
for i in values:
    if i % 2 != 0:                # i%2 outputs remainder when divided by 2                                            
        continue                  # 'continue' will skip odd numbers and move to the next iteration
    print i/2                                    

# WHILE IF BREAK: The break statement halts the execution of the loop

command_list = ['start', 'process', 'process', 'process', 'stop', 'start', 'process', 'stop']
while command_list:
    command = command_list.pop(0)
    if command == 'stop':
        break                     # when you hit the 5th element, the loop breaks
    print(command)
    
# WHILE TRUE: 
command_list = ['start', 'process', 'process', 'process', 'stop', 'start', 'process', 'stop']
while command_list:
    command = command_list.pop(0)
    if command == 'stop':
        print(command_list)

# FOR IF BREAK: Example of using break with 'for' statements
values = [7, 6, 4, 7, 19, 2, 1]
for x in values:
    if x <= 10:
        if x > 10:
            break               # the loop breaks as soon as it encounters a number > 10
        print 'Found:', x
        
#############################################################################################################
# 3. List Comprehension
#############################################################################################################
# List comprehensions are a powerful way to generate lists using the for/in and if keywords. In the example below, a list is created for cubes of number from 1 - 10, but only of the cubes are divisible by 4. 

cubes_by_four = [i**3 for i in xrange(1,11) if (i**3) % 4 == 0]
print cubes_by_four

#############################################################################################################
# 4. Lambda Functions
#############################################################################################################
# One of the more powerful aspects of Python is that it allows for a style of programming called functional programming, which means that you're allowed to pass functions around just as if they were variables or values. Only we don't need to actually give the function a name; it does its work and returns a value without one. That's why the function the lambda creates is an anonymous function.

my_list = range(16)
print filter(lambda x: x % 3 == 0, my_list)

# Using list comprehension with a lambda function
squares=[x**2 for x in range(1,11)]
print filter(lambda x: x in range(30,71),squares)

#############################################################################################################
# 5. Classes & Methods
#############################################################################################################

# Python is an object-oriented programming language, which means it manipulates programming constructs called 'objects'. You can think of an object as a single data structure that contains data as well as functions; functions of objects are called 'methods'.

# __init__: This function is required for classes, and it's used to initialize the objects it creates. __init__() always takes at least one argument, self, that refers to the object being created. You can think of __init__() as the function that "boots up" each object the class creates.

class Fruit(object):
    """A class that makes various tasty fruits."""
    def __init__(self, name, color, flavor, poisonous):
        self.name = name
        self.color = color
        self.flavor = flavor
        self.poisonous = poisonous

    def description(self):
        print "I'm a %s %s and I taste %s." % (self.color, self.name, self.flavor)

    def is_edible(self):
        if not self.poisonous:
            print "Yep! I'm edible."
        else:
            print "Don't eat me! I am super poisonous."

lemon = Fruit("lemon", "yellow", "sour", False)

lemon.description()
lemon.is_edible()

#############################################################################################################
# 6. Programmatically downloading and unzipping files
############################################################################################################# 
import zipfile
import urllib2
import os
import pandas as pd
import numpy as np

source_url = 'ftp://ftp.nhtsa.dot.gov/GES/GES12/GES12_Flatfile.zip'
zip_name = 'GES12_Flatfile.zip'
cwd = os.getcwd()
dir_path  = os.path.join(cwd, 'GES2012')
zip_path = os.path.join(dir_path, zip_name)

# We'll make a directory for you to play around with,
# then when you're done playing you can just delete the directory
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Download the file from GES website if you haven't already
if not os.path.exists(zip_path):
    response = urllib2.urlopen(source_url)
    with open(zip_path, 'wb') as fh:
        x = response.read()
        fh.write(x)

# Extract all the files from that zipfile
with zipfile.ZipFile(os.path.join(dir_path, zip_name), 'r') as z:
    z.extractall(dir_path)
    
#See what we just unzipped
os.listdir(dir_path)

# Extract raw data
cwd = os.getcwd()
dir_path  = os.path.join(cwd, 'GES2012')
input_file_path = os.path.join(dir_path, 'PERSON.TXT')

























