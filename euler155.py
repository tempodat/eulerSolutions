from fractions import Fraction

res = [{Fraction(1)}]

series = lambda a,b: (a*b)/(a+b)
parallel = lambda a,b: a+b

# take two sets of resistance values, and return a set of values one can
# reach by combining a resistor from set A with a resistor from set B, either 
# in series or in parallel. 
def combined_resistances(res1: set, res2: set) -> set:
    values = set()
    for a in res1:
        for b in res2:
            values.add(series(a,b))
            values.add(parallel(a,b))
    return values

# populate the array res. nth value of res should be equal to the set of possible
# resistances with n resistors (only connecting in series/parallel, no bridge type circuits).
# calculates each successive set of possible values by combining previous sets of resistance 
# values possible to reach with smaller circuits, so for example if we're calculating the reachable
# values for n=3 resistors, we'll look at every pair of a resistor made with 1 and a resistor made
# with 2, and get all of their possible composite circuit's resistances.
def evaluate(n):
    global res
    if len(res) >= n:
        return
    elif len(res) < n - 1:
        for i in range(len(res),n):
            evaluate(i)
			
    values = set()
    for n1 in range(1,n // 2 + 1):
        n2 = n - n1
        result = combined_resistances(res[n1-1],res[n2-1])
        values.update(result)
    res.append(values)

# populate array, and print number of unique resistance values with <= 18
N = 18
evaluate(N)
print(len(set.union(*res[:N])))

# for N=18 on my potato laptop:
# CPU times: total: 5min 15s
# Wall time: 5min 41s
