#!/usr/bin/env python
# coding: utf-8

# # Python Utilities for Project Euler
# 
# <div style="text-align: right">Peter Norvig, 20 Jan 2017</div>
# <div style="text-align: right">tempodat, 17 Jun 2024</div>
# 
# 
# After showing my utilities for [Advent of Code](http://adventofcode.org), I got some feedback:
# 
#  1. Some of these are recipes in the `itertools` module (with different names).
#  2. What about  utilities for [Project Euler](https://projecteuler.net/about)?
#  
# My answers: 
# 
#  1. I agree! I have renamed some of my utilities to be consistent with the `itertools` recipes.
#  2. Here you go.

# # Imports
# 
# In Project Euler I am writing short programs for my own consumption, so brevity is important, and I use `"from"` imports more often than I normally would:

# In[1]:


from collections import defaultdict, deque, Counter, namedtuple, abc
from fractions   import Fraction
from functools   import lru_cache, wraps
from itertools   import chain, cycle, islice, combinations, permutations, repeat, takewhile, zip_longest
from itertools   import product as crossproduct, count as count_from
from math        import ceil, floor, factorial, gcd, log, sqrt, inf
from tqdm        import tqdm
from bisect import bisect_right
import random
import time


# # Utilities
# 
# Here are the general utility functions (and data objects) I define:

# In[2]:


million  = 10 ** 6      # 1,000,000
Ø        = frozenset()  # Empty set
distinct = set          # Function to return the distinct elements of a collection of hashables
identity = lambda x: x  # The function that returns the argument
cat      = ''.join      # Concatenate strings

def first(iterable, default=False):
    "Return the first element of an iterable, or default if it is empty."
    return next(iter(iterable), default)

def first_true(iterable, pred=None, default=None):
    """Returns the first truthy value in the iterable.
    If no true value is found, returns *default*
    If *pred* is not None, returns the first item
    for which pred(item) is truthy."""
    # first_true([a,b,c], default=x) --> a or b or c or x
    # first_true([a,b], fn, x) --> a if fn(a) else b if fn(b) else x
    return next(filter(pred, iterable), default)

def count_under_value(lst, item):
    """docs are hard"""
    i = bisect_left(lst, item)
    if i != len(alist):
        return i
    raise ValueError


def upto(iterable, maxval):
    "From a monotonically increasing iterable, generate all the values <= maxval."
    # Why <= maxval rather than < maxval? In part because that's how Ruby's upto does it.
    return takewhile(lambda x: x <= maxval, iterable)

# def onward(iterable, minval):
#     "From a monotonically increasing iterable, generate all the values >= minval."
#     p

def multiply(numbers):
    "Multiply all the numbers together."
    result = 1
    for n in numbers:
        result *= n
    return result

def transpose(matrix): return tuple(zip(*matrix))

def isqrt(n):
    "Integer square root (rounds down)."
    return int(n ** 0.5)

def ints(start, end):
    "The integers from start to end, inclusive. Equivalent to range(start, end+1)"
    return range(start, end+1)

def groupby(iterable, key=identity):
    "Return a dict of {key(item): [items...]} grouping all items in iterable by keys."
    groups = defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)
    return groups

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks:
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def overlapping(iterable, n):
    """Generate all (overlapping) n-element subsequences of iterable.
    overlapping('ABCDEFG', 3) --> ABC BCD CDE DEF EFG"""
    if isinstance(iterable, abc.Sequence):
        yield from (iterable[i:i+n] for i in range(len(iterable) + 1 - n))
    else:
        result = deque(maxlen=n)
        for x in iterable:
            result.append(x)
            if len(result) == n:
                yield tuple(result)
                
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    return overlapping(iterable, 2)
                
def sequence(iterable, type=tuple):
    "Coerce iterable to sequence: leave it alone if it is already a sequence, else make it of type."
    return iterable if isinstance(iterable, abc.Sequence) else type(iterable)

def join(iterable, sep=''):
    "Join the itemsin iterable, converting each to a string first."
    return sep.join(map(str, iterable))

def grep(pattern, lines):
    "Print lines that match pattern."
    for line in lines:
        if re.search(pattern, line):
            print(line)
            
def nth(iterable, n, default=None):
    "Returns the nth item (or a default value)."
    return next(islice(iterable, n, None), default)

def ilen(iterable):
    "Length of any iterable (consumes generators)."
    return sum(1 for _ in iterable)

def quantify(iterable, pred=bool):
    "Count how many times the predicate is true."
    return sum(map(pred, iterable))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    seq = sequence(iterable)
    return flatten(combinations(seq, r) for r in range(len(seq) + 1))

def shuffled(iterable):
    "Create a new list out of iterable, and shuffle it."
    new = list(iterable)
    random.shuffle(new)
    return new
    
flatten = chain.from_iterable

def int_cache(f):
    """Like lru_cache, but designed for functions that take a non-negative integer as argument;
    when you ask for f(n), this caches all lower values of n first. That way, even if f(n) 
    recursively calls f(n-1), you will never run into recursionlimit problems."""
    cache = [] # cache[i] holds the result of f(i)
    @wraps(f)
    def memof(n):
        for i in ints(len(cache), n):
            cache.append(f(i))
        return cache[n]
    memof._cache = cache
    return memof


# # Primes
# 
# My class `Primes` does what I need for the many Project Euler problems that involve primes:
# 
# * Iterate through the primes up to 2 million.
# * Instantly check whether an integer up to 2 million is a prime.
# * With a bit more computation, check if, say, a 12-digit integer is prime.
# 
# I precompute the primes up to 2 million, using 
# a [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), and then cache
# the primes as both a list (to iterate through) and a set (to check membership). If there are `n`
# primes currently cached and you ask for `primes[n+1]` (either directly, or indirectly by iterating over `primes`), 
# then the cache will be automatically doubled in size. But if you just ask if, say, "`123456789011 in primes`",
# then I use repeted trial division without extending the cache.

# In[3]:


class Primes:
    """Given `primes = Primes(2 * million)`, we can do the following:
    * for p in primes:                  # iterate over infinite sequence of primes
    * 37 in primes => True              # primality test
    * primes[0] => 2, primes[1] => 3    # nth prime
    * primes[:5] => [2, 3, 5, 7, 11]    # first 5 primes
    * primes[5:9] => [13, 17, 19, 23]   # slice of primes 
    * primes.upto(10) => 2, 3, 5, 7     # generate primes less than or equal to given value"""

    def __init__(self, n):
        "Create an iterable generator of primes, with initial cache of all primes <= n."
        # sieve keeps track of odd numbers: sieve[i] is True iff (2*i + 1) has no factors (yet) 
        N = n // 2 # length of sieve
        sieve = [True] * N
        try:
            description = f"upgr prime db from {self.maxn} to {n}"
        except AttributeError:
            description = f"init prime db at {n}"
        for i in tqdm(range(3, isqrt(n) + 1, 2), desc=description):
            if sieve[i // 2]: # i is prime
                # Mark start, start + i, start + 2i, ... as non-prime
                start = i ** 2 // 2
                sieve[start::i] = repeat(False, len(range(start, N, i)))
        self._list = [2] + [2*i+1 for i in range(1, N) if sieve[i]]
        self._set  = set(self._list)
        self.maxn  = n # We have tested for all primes < self.maxn

    def __contains__(self, n):
        "Is n a prime?"
        # If n is small, look in _set; otherwise try prime factors up to sqrt(n)
        if n <= self.maxn:
            return n in self._set
        else:
            return not any(n % p == 0 for p in self.upto(n ** 0.5))

    def __getitem__(self, index):
        "Return the ith prime, or a slice: primes[0] = 2; primes[1] = 3; primes[1:4] = [3, 5, 7]."
        stop = (index.stop if isinstance(index, slice) else index)
        if stop is None or stop < 0:
            raise IndexError('Number of primes is infinite: https://en.wikipedia.org/wiki/Euclid%27s_theorem')
        while len(self._list) <= stop:
            # If asked for the ith prime and we don't have it yet, we will expand the cache.
            self.__init__(2 * self.maxn)
        return self._list[index]

    def __len__(self):
        """ Return the number of primes cached """
        return len(self._list)

    def totient(self, n):
        """ Return number of primes upto n """
        if self.maxn < n:
            self.__init__(max(n, 2 * self.maxn))
        return bisect_right(self._list, n)
        
    def upto(self, n):
        "Yield all primes <= n."
        if self.maxn < n:
            self.__init__(max(n, 2 * self.maxn))
        return upto(self._list, n)

    def between(self, a, b):
        "Yield all primes a <= p <= b"
        if self.maxn < b:
            self.__init__(max(n, 2 * self.maxn))
        a_index = self.totient(a)
        b_index = self.totient(b)
        return iter(self._list[a_index:b_index])

# There are 148,933 primes under 2 million, which is a small enough number that I'm not concerned with the memory consumed by `._list` and `._set`. If I needed to store 100 million primes,  I would make different tradeoffs. For example, instead of a list and a set, I would probably just keep `sieve`, and make it be an `array('B')`. This would take less space (but for "small" sizes like 2 million, the current implementation is both faster and simpler).
# 
# 
# # Factors
# 
# Project Euler also has probems about prime factors, and divisors. I need to:
# 
# * Quickly find the prime factors of any integer up to a million.
# * With a bit more computation, find the prime factors of a 12-digit integer.
# * Find the complete factorization of a number.
# * Compute Euler's totient function.
# 
# I will cache the factors of all the integers up to a million.  To be more precise, I don't actually keep a list of all the factors of each integer; I only keep the largest prime factor. From that, I can easily compute all the other factors by repeated division. If asked for the factors of a number greater than a million, I do trial division until I get it under a million. In addition, `Factors` provides `totient(n)` for computing [Euler's totient function](https://en.wikipedia.org/wiki/Euler's_totient_function), or Φ(n), and `ndivisors(n)` for the total [number of divisors](http://primes.utm.edu/glossary/xpage/tau.html) of `n`.

# In[4]:


class Factors:
    """Given `factors = Factors(million)`, we can do the following:
    * factors(360) => [5, 3, 3, 2, 2, 2]  # prime factorization
    * factors.largest[360] => 5           # largest prime factor
    * distinct(factors(360)) => {2, 3, 5} # distinct prime factors
    * factors.ndivisors(28) => 6          # How many positive integers divide n?
    * factors.totient(36) => 12           # How many integers below n are relatively prime to n?"""
    def __init__(self, maxn):
        "Initialize largest[n] to be the largest prime factor of n, for n < maxn."
        self.largest = [1] * maxn
        for p in tqdm(primes.upto(maxn), total=primes.totient(maxn), desc=f"init factors up to {maxn}"):
            self.largest[p::p] = repeat(p, len(range(p, maxn, p)))
            
    def ndivisors(self, n):
        "The number of divisors of n."
        # If n = a**x * b**y * ..., then ndivisors(n) = (x+1) * (y+1) * ...
        exponents = Counter(self(n)).values()
        return multiply(x + 1 for x in exponents)
        
    def totient(self, n):
        "Euler's Totient function, Φ(n): number of integers < n that are relatively prime to n."
        # totient(n) = n∏(1 - 1/p) for p ∈ distinct(factors(n))
        return int(n * multiply(1 - Fraction(1, p) for p in distinct(self(n))))
      
    def __call__(self, n):
        "Return a list of the numbers in the prime factorization of n."
        result = []
        # Need to make n small enough so that it is in the self.largest table
        if n >= len(self.largest):
            for p in primes:
                while n % p == 0:
                    result.append(p)
                    n = n // p
                if n < len(self.largest):
                    break
        # Now n is in the self.largest table; divide by largest[n] repeatedly:
        while n > 1:
            p = self.largest[n]
            result.append(p)
            n = n // p
        return result
    

# # Tests
# 
# Here are some unit tests (which also serve as usage examples):

# In[5]:

if __name__ == '__main__':
    get_ipython().run_line_magic('time', 'primes  = Primes(2 * million)')
    factors = Factors(million)

    def tests():
        global primes, factors
        primes = Primes(2 * million)
        factors = Factors(million)
        
        assert first('abc') == first(['a', 'b', 'c']) == 'a'
        assert first(primes) == 2
        assert cat(upto('abcdef', 'd')) == 'abcd'
        assert multiply([1, 2, 3, 4]) == 24
        assert transpose(((1, 2, 3), (4, 5, 6))) == ((1, 4), (2, 5), (3, 6))
        assert isqrt(9) == 3 == isqrt(10)
        assert ints(1, 100) == range(1, 101)
        assert identity('anything') == 'anything'
        assert groupby([-3, -2, -1, 1, 2], abs) == {1: [-1, 1], 2: [-2, 2], 3: [-3]}
        assert sequence('seq') == 'seq'
        assert sequence((i**2 for i in range(5))) == (0, 1, 4, 9, 16)
        assert join(range(5)) == '01234'
        assert join(range(5), ', ') == '0, 1, 2, 3, 4'
        assert cat(['do', 'g']) == 'dog'
        assert nth('abc', 1) == nth(iter('abc'), 1) == 'b'
        assert quantify(['testing', 1, 2, 3, int, len], callable) == 2 # int and len are callable
        assert quantify([0, False, None, '', [], (), {}, 42]) == 1  # Only 42 is truish
        assert set(powerset({1, 2, 3})) == {(), (1,), (1, 2), (1, 2, 3), (1, 3), (2,), (2, 3), (3,)}
        assert first_true([0, None, False, {}, 42, 43]) == 42
        assert list(grouper(range(8), 3)) == [(0, 1, 2), (3, 4, 5), (6, 7, None)]
        assert list(pairwise((0, 1, 2, 3, 4))) == [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert list(overlapping((0, 1, 2, 3, 4), 3)) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
        assert list(overlapping('abcdefg', 4)) == ['abcd', 'bcde', 'cdef', 'defg']    
        @int_cache
        def fib(n): return (n if n <= 1 else fib(n - 1) + fib(n - 2))
        f = str(fib(10000))
        assert len(f) == 2090 and f.startswith('33644') and f.endswith('66875')
    
        assert 37 in primes
        assert primes[0] == 2 and primes[1] == 3 and primes[10] == 31
        assert primes[:5] == [2, 3, 5, 7, 11]
        assert primes[5:9] == [13, 17, 19, 23]
        assert 42 not in primes
        assert 1299721 in primes
        assert million not in primes
        assert (2 ** 13 - 1) in primes
        assert (2 ** 31 - 1) in primes
        assert list(primes.upto(33)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        assert primes.maxn == 2 * million # Make sure we didn't extend cache
        assert len(primes._set) == len(primes._list) == 148933
        
        assert factors(720) == [5, 3, 3, 2, 2, 2, 2]
        assert distinct(factors(720)) == {2, 3, 5}
        assert factors(37) == [37]
        assert distinct(factors(72990720)) == {2, 3, 5, 11}
        assert factors.ndivisors(6) == 4
        assert factors.ndivisors(28) == 6
        assert factors.ndivisors(720) == 30
        assert factors.largest[720] == 5
        assert factors.totient(36) == 12
        assert factors.totient(43) == 42
        for n in (28, 36, 37, 99, 101): 
            assert list(primes.upto(n)) == list(upto(primes, n))
            assert factors.totient(n) == quantify(gcd(n, d) == 1 for d in ints(1, n))
            assert n == sum(factors.totient(d) for d in ints(1, n) if n % d == 0)
    
    
    
        return 'pass'
    print(tests())


# # In[6]:


# # Instantiate both primes and factors
# get_ipython().run_line_magic('time', 'primes = Primes(2 * million)')
# get_ipython().run_line_magic('time', 'factors = Factors(million)')


# # In[7]:


# # Check primality for numbers in cache
# get_ipython().run_line_magic('time', '1000003 in primes')


# # In[7]:


# get_ipython().run_line_magic('time', '1000001 in primes')


# # In[ ]:


# # Check primality for numbers beyond the cache
# get_ipython().run_line_magic('time', '2000003 in primes')


# # In[ ]:


# # Factor numbers in cache
# get_ipython().run_line_magic('time', 'factors(98765)')


# # In[ ]:


# get_ipython().run_line_magic('time', 'factors(810000)')


# # In[ ]:


# # Factor numbers beyond the cache
# get_ipython().run_line_magic('time', 'factors(74843 ** 2)')


# # In[ ]:


# x = 1000003 ** 3 * 1999993 ** 5
# print(x)
# get_ipython().run_line_magic('time', 'factors(x)')


# # In[ ]:


# get_ipython().run_line_magic('time', 'sum(primes.upto(million))')


# # In[ ]:


# # sum of the first 100,000 primes
# get_ipython().run_line_magic('time', 'sum(primes[:100000])')


# # In[ ]:


# # First prime greater than a million
# get_ipython().run_line_magic('time', 'first(p for p in primes if p > million)')


# # In[ ]:


# # sum of the integers up to 10,000 that have exactly 3 distinct factors
# get_ipython().run_line_magic('time', 'sum(n for n in range(1, 10000) if len(distinct(factors(n))) == 3)')


# # In[ ]:


# # sum of the integers up to 10,000 that have exactly 3 divisors
# get_ipython().run_line_magic('time', 'sum(n for n in range(1, 10000) if factors.ndivisors(n) == 3)')


# # In[ ]:


# # The sum of the totient function of the integers up to 1000 
# get_ipython().run_line_magic('time', 'sum(map(factors.totient, range(1, 10000)))')


# # Project Euler Regression Testing
# 
# My strategy for managing solutions to problems, and doing regression tests on them:
# * My solution to problem 1 is the function `problem_1()`, which returns the solution when called (and so on for other problems).
# * Once I have verified the answer to a problem (checking it on the Project Euler site), I store it in a dict called `solutions`.
# * Running `verify()` checks that all `problem_`*n* functions  return the correct solution. 
# 
# Project Euler asks participants not to publish solutions to problems, so I will comply, and instead show the solution to  three fake problems:

# In[ ]:


# def problem_1(N=100):
#     "Sum of integers: Find the sum of all the integers from 1 to 100 inclusive."
#     return sum(ints(1, N))

# def problem_2(): 
#     "Two plus two: how much is 2 + 2?"
#     return int('2' + '2')

# def problem_42():
#     "What is life?"
#     return 6 * 7

# solutions = {1: 5050, 2: 4} 


# def verify(problem_numbers=range(1, 600)):
#     """Main test harness function to verify problems. Pass in a collection of ints (problem numbers).
#     Prints a message giving execution time, and whether answer was expected."""
#     print('Num       Time Status Answer           Problem Description   Expected')
#     print('=== ========== ====== ================ ===================== ========')
#     for p in problem_numbers:
#         name = 'problem_{}'.format(p)
#         if name in globals():
#             fn     = globals()[name]
#             t0     = time.time()
#             answer = fn()
#             t      = time.time() - t0
#             desc   = (fn.__doc__ or '??:').split(':')[0]
#             status = ('NEW!'   if p not in solutions else 
#                       'WRONG!' if answer != solutions[p] else
#                       'SLOW!'  if t > 60 else
#                       'ok')
#             expected = (solutions[p] if status == 'WRONG!' else '')
#             print('{:3d} {:6.2f} sec {:>6} {:<16} {:<21} {}'
#                   .format(p, t, status, answer, desc, expected))

# verify()


# # Custom Code


