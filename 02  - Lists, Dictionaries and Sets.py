##############################################################################################################
# 1.Lists & Tuples
#############################################################################################################

a = [10, 11, 12, 13, 14]           # List in Python are an ordered sequence of any kind of object
print a * 2                        # lists are not vectorized
range(5)                           # create a list of 5 elements
range(2, 7)                        # create a list from 2 to 6
range(2, 7, 2)                     # create a list between 2 to 6 with a step value of 2
a[1:3] = [10.5, 11.25, 11.5]       # use indexing to replace elements in the list
len(a)                             # length of list
a.append(15)                       # a.append() method adds a new element at the end of the list (one element)
a.extend([16, 17])                 # use the a.extend method to add >1 elements at the end of the list
a.count(10)                        # count() method tells you how many of a certain element are in a list
a.pop(2)                           # pop() method will 'pop' out the 2nd element from the list (default is last element)
a.remove(10)                       # remove() method will remove the specified element from the list
del a[4]                           # alternately use the del command along with the index

print sorted(a)                    # sorted() sorts from lowest to highest
a.reverse()                        # the reverse() method reverses the elements of the list in-place
b = (1, 2, 3, 4, 5)                # a tuple is an immutable list - elements cannot be replaced via indexing

# INDEXING LOOPS WITH STRIDE: the proper indexing structure is: [start:end:stride]. Stride is the size of every 'index step'
my_list = range(1, 11) 
print  my_list[0::2]               # with a stride of 2, this will print all the odd numbers
print my_list[-2::-2]              # negitive stride to move backwards starting at 9


#############################################################################################################
# 2. Dictionaries
#############################################################################################################
# Each entry is a key-value pair, where the keys are the words, and the values are the definitions. With multiple definitions, you might instead have a list of definitions instead of a single definition for the value, but the idea is the same.'''

woobsters = {}                               # initialize a dictionary using curly brackets
woobsters['mutable'] = 'liable to change'    # adding key-value pairs
woobsters['flow'] = 'for loops'
print woobsters['flow']                      # lookup value of a key (crude method)
print woobsters.get('flow')                  # use the get method to return values (more refined)
print woobsters.get('jars', 'unknown')       # the second optional argument for what to return if no key exists
woobsters.pop('mutable', 0)                  # pops the mutable key, second argument is optional like above
print woobsters
woobsters.keys()                             # lists all keys as a list
woobsters.values()                           # lists all values as a list
woobsters.items()                            # lists all key value pairs
del woobsters['mutable']                     # use the del function 

synonyms = {}
synonyms['mutable'] = ['changeable',         # you can have multiple values for one key
                       'variable',
                       'varying',
                       'fluctuating']
synonyms['immutable'] = ['fixed',
                         'set',
                         'rigid']
print synonyms

inventory = dict([('foozelator', 123),       # use the dict function to create dictionaries
                  ('frombicator', 18),
                  ('spatzleblock', 34),
                  ('snitzelhogen', 23)])
print inventory

inventory['frombicator'] += 1                # you can increment values if they are numbers
print inventory

#############################################################################################################
# 3. Sets
#############################################################################################################
# Sets are collections of things like lists but unlike lists, sets are unordered
email_list = ['joe@acme.com', 'sue@cooks.com', 'bud@plumbers.com',
              'bugs@cartoon.com', 'betty@cartoon.com',
              'joe@acme.com', 'bugs@cartoon.com']
              
# Sets are useful for removing duplicates, because Python removes any duplicates in a set. Also don't confuse sets with
# dictionaries because Python represents sets with curly brackets.
emails = set(email_list)
print emails

a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
union = a | b                    # set union - like an OR operator
intersection = a & b             # set intersection - like an AND operator
difference = a - b               # set difference - elements in a and not in b
symmetric_difference = a ^ b     # elements which are in a or b but not in both - like a XOR operator
b.issubset(a)                    # use issubset to test if b is a subset of a
a.issuperset(b)                  # use issuperset to test if a is a super-set of b
a.isdisjoint(b)                  # use isdisjoint to test of a and b have nothing in common
a.add(16)                        # to add a single element into a set
a.update([13, 14, 15])           # to add more than one element into a set
a.remove(15)                     # removes a single element from a set
a.pop()                          # pops an element from a set (you don't care which one)
a.discard(1000)                  # removes an element; if element does not exist it will not cause an error

# FROZEN PAIRS are special sets that are immutable and that can be used as keys to a dictionary
LA_NY = frozenset(['LA', 'NY'])              # define frozen set with a list of two city pairs
distances = {}                               # initialize a dictionary
distances[LA_NY] = 2498                      # assign the first kv pair
distances[frozenset(['SEA', 'LA'])] = 1500   # create a new kv pair
distances.get(frozenset(['SEA', 'LA']))      # query the dictionary - you need to use 'frozenset' with 'get'