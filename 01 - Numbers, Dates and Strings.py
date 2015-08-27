#############################################################################################################
# 1. Utilities
#############################################################################################################
a = 10
b = 'string'
c = [1, 2, 3, 4]
type(a)                                     # use type to find out the type of object
len(c)                                      # string length function
id(b)                                       # Python asssigns a unique identifier to every object

#############################################################################################################
# 2. Numbers and Math Operators
#############################################################################################################
# Basic Math Functions   
  
# in Python 2.x, division does not yield integers! So you have two options, use __future__ or use floating point numbers
print 5/2                                   # yields 2!
print 5/float(2)                            # yields 2.5
from __future__ import division             # imports Python 3.x division function
print 5/2                                   # yields 2.5
                                                  
abs(-3)                                     # absolute function
round(2.718281828)                          # rounding up
max(0, min(10, -1, 4, 3))                   # max and min functions
int(2.718281828)                            # converts to integer (2)
float(2)                                    # converts to floating point (2.0)
exponent = 10**2                            # use '**' for exponents
10%3                                        # modulo (%) outputs the remainder on division
                   
#the math module is required for certain functions for eg: sin, cos, tan, log,log10, pi etc. 
import math as math  
print dir(math)                             # this prints all available functions in the math module 
print math.sqrt(100)
# It is bad practice however, to invoke the entire module. 
from math import sqrt
print sqrt(81)

#############################################################################################################
# 3. Date & Time formats
#############################################################################################################
from datetime import datetime                 # Python uses the datetime module
now = datetime.now()                          # outputs current date/time in UNIX timestamp format
year = now.year
month = now.month
day = now.day
print 'Today is: %s/%s/%s' % (day,month,year) # use string formatting to display desired format


#############################################################################################################
# 4. Intro to Strings
#############################################################################################################
# Beware of Indentations in Python!
# Unlike many other languages, Python uses the indentation in the source code for interpretation.
# So for instance, for the following two scripts are different!

if 0 == 1:
    print 'We are in a world of arithmetic pain'
print 'Thank you for playing'

if 0 == 1:
    print 'We are in a world of arithmetic pain'
    print 'Thank you for playing'

# Basic Functions
s = 'hello world '
space = ' '
print(s * 3)                                # strings can be multiplied by numbers
print "Hello" + "Jason"                     # + for concatenation
word_list = s.split()                       # s.split splits a string into a list based on white space
space.join(word_list)                       # joins word lists based on the space variable
s.replace("world", "Mars")                  # s.replace method replaces words in a list
s.upper()                                   # converts to uppercase
s.strip()                                   # strip() method removes excess characters at the end of a string
dir(s)                                      # dir() function lists all the methods on an object (string in this case)
print("""hello
world""")                                   # triple quotes for strings with line breaks

# Special Characters & Escapes
print "line\nbreak"                         # \n for line break escape sequence
print "\tthis is a tabbed line"             # \t for tab escape sequence
print "I am 6\'2\" tall"                    # \' or \" to include single or double quotes in a string
print """This is a list:
\t* Item 1 \t* Item 2"""                    # \t to create lists

# Number/String Conversions
str(1.1 + 2.2)                              # convert numbers to string
hex(255)                                    # hexadecimal representation
oct(255)                                    # octal representation
bin(5)                                      # binary representation
int('23')                                   # convert string to integer
float('23.45')                              # convert string to floating point number

# String Indexing
# Indexing is 0-based instead of 1-based, so the index 0 gives the first character. When thinking about indices,
#   it's useful to think of them as being between the elements, rather than on the elements.
s = '123456789'
sentence = 'My name is Jason Macwan and I want to learn Python'
print s[0]                                   # [0] indexes the 1st element
print s[4]                                   # indexes the 5th element
print s[-3]                                  # indexes backward -- the 7th element in this case
print s[1:3]                                 # range indexing everything between the 1st and 3rd element
print s[1:-4]                                # range indexing from front and back
print s[:3]                                  # index up to the 3rd element
print s[-3:]                                 # index up to the -3rd element

# the find method outputs the start position of the expression. The number (optional tells Python from which position to start searching. If the search expression is not found, Python will return -1
sentence.find('Macwan', 10)      

#############################################################################################################
# 5. Formatting Strings
#############################################################################################################
# String Formatting the old way using %
first = 'Jason'
last = 'Macwan'
print 'Name: %s %s.' % (first, last)
print 'My name is %s.' % first               # no brackets required for one string

# String Formatting the new way using str.format
# Accessing arguments by position:
'{0}, {1}, {2}'.format('a', 'b', 'c')        # 0 position is 'a', so on and so forth
'{}, {}, {}'.format('a', 'b', 'c')           # 2.7+ only
'{2}, {1}, {0}'.format('a', 'b', 'c')
'{2}, {1}, {0}'.format(*'abc')               # unpacking argument sequence
'{0}{1}{0}'.format('abra', 'cad')            # arguments' indices can be repeated
# Accessing arguments by name:
'Coordinates: {latitude}, {longitude}'.format(latitude='37.24N', longitude='-115.81W')

# Aligning the text and specifying a width:
'{:<30}'.format('left aligned')              # left aligned (:<), padding of 30
'{:>30}'.format('right aligned')             # right aligned (:>), padding of 30
'{:^30}'.format('centered')                  # centered (:^), padding of 30
'{:*^30}'.format('centered')                 # use '*' as a whitespace filler




        

  

    
        
    