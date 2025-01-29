from eulerlib import *

@int_cache
def ffactors(n):
    return sum((Counter(factors(i)) for i in range(1,n+1)), start=Counter())

@lru_cache
def cfactors(n,k):
    return ffactors(n)-ffactors(k)-ffactors(n-k)

def mul(counter):
    return multiply(p**counter[p] for p in counter)


N = 51 # rows

is_squarefree = lambda counter: all(exp == 1 for exp in counter.values())

# factorizations = set(cfactors(n,k) for k in range(1,N) for n in range(k,N))
# squarefree = filter(is_squarefree, factorizations)
# sumofsquarefree = sum(map(mul, squarefree)) // 2
# print(sumofsquarefree)

seen = set()
checked = set()
for k in range(1,N):
    for n in range(k,N):
        fac = cfactors(n,k)
        val = mul(fac)
        
        if val in seen:
            continue
        seen.add(val)

        if all(exp == 1 for exp in fac.values()):
            checked.add(val)

print(sum(checked))
