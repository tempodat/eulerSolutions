import numpy as np
from eulerlib import *

primes = Primes(20 * million)

sevenseg = np.array([[1,1,1,0,1,1,1],
                     [0,0,1,0,0,1,0],
                     [1,0,1,1,1,0,1],
                     [1,0,1,1,0,1,1],
                     [0,1,1,1,0,1,0],
                     [1,1,0,1,0,1,1],
                     [1,1,0,1,1,1,1],
                     [1,1,1,0,0,1,0],
                     [1,1,1,1,1,1,1],
                     [1,1,1,1,0,1,1]],dtype=int)

savingsmatrix = 2 * sevenseg @ sevenseg.T # it's magical how this works

def digit_sum(n):
    return sum(map(int,str(n)))

def digits(n):
    if n > 9:
        d, m = divmod(n,10)
        yield m
        yield from digits(d)
    else:
        yield n

def savings_inefficient(n, table):
    while n > 9:
        dsum = digit_sum(n)
        for d1, d2 in zip(digits(n), digits(dsum)):
            table[d1,d2] += 1
        n = dsum
    
SNC = 100 # Samll Number cutoff point

savings_memo = np.zeros((SNC,10,10),dtype=int)
for i in range(SNC):
    savings_inefficient(i, savings_memo[i,:,:] )


total = np.zeros((10,10),dtype=int)
digitsums = np.zeros(SNC,dtype=int)

our_primes = list(primes.between(10 * million,20 * million))
for p in tqdm(our_primes):
    dsum = digit_sum(p)
    for d1, d2 in zip(digits(p), digits(dsum)):
        total[d1,d2] += 1
    digitsums[dsum] += 1

total += np.tensordot(savings_memo, digitsums, axes = ((0),(0)))
print(np.sum((total * savingsmatrix)))