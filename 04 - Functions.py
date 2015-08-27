#############################################################################################################
# 1. Functions
#############################################################################################################

# In order to use a procedure, you need the name of the procedure using def, followed by a left parenthesis, a list of the procedure's inputs (sometimes called operands or arguments)

def abbaize(a,b):               # defines a function 'abbaize' with two inputs
    c = a+b*2+a                 # concatenates the two inputs
    return c                    # return prints the output
    
# remember with functions, return outputs to a new line - especially important if you need one function's output to feed into another function
    
def distance_from_zero(n):
    if type(n) == int or type(n) == float:
        print "Print does not output the function"
        return abs(n)          
    else:
        return "Nope"