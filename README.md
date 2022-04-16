# Regressi

Simple package to do linear regressions

## Regressi implementation

This is the full package, with all the functionalities

````python
from Regressi import *

# Simple call with default settings
regressi(list, list, delta_list, delta_list, format="y=ax+b", iterations=1000, graph=True)

# Get the result of the linear regression
r = regressi(list, list, delta_list, delta_list, format="y=ax+b", iterations=1000, graph=False) # graph can be True

print(r)
# >> a±Δa, b±Δb

r[0]
# >> a
''' Index:
	0: a,
	1: b,
	2: Δa,
	3: Δb
'''

r2 = regressi(list, list, delta_list, delta_list, format="y=ax+b", iterations=1000, graph=False) # graph can be True

r.infof(r2) # r.overof(r2)
# return True if whatever x is, x(a-a2) < b2-b, else return False

r((min, max), precision)
# return ax+b list for all x in [min, max] within a precision choosed

'''
For the next example, I assume that a file is created like:

1;2
2;3
...
'''

l1, l2 = fileHandler.cread("filename.txt")
regressi(l1, l2, delta_list, delta_list, format="y=ax+b", iterations=1000, graph=True)

'''
For this example, I assume that a linear regression was already calculate, and named r
'''

fileHandler.cwrite(r, (min, max), precision)
# Write a file with the values of ax+b for all x in [min, max] within a precision choosed
````

## Simple Regressi

There is an easier package of regressi, just made to create graphs of the linear regression

````python
from Regressi_simple import *

# Call with default settings
regressi(list, list, delta_list, delta_list, format="y=ax+b", iterations=1000)
````

## Any questions ?

Feel free to ask us anything you want on discussions and pull requests
