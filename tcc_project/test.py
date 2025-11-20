import numpy as np
from time import time



l = [i for i in range(10000000)]
x = time()
for i in l:
    if i % 2 ==0:
        i = i + 1
    else:
        i = i - 1
print(time() - x)


def func(i):
    if i % 2 ==0:
        i =i + 1
    else:
        i= i - 1
    return i



l2 = np.arange(10000000)
l2.mean()
ufunc = np.frompyfunc(func, 1, 1)
x = time()
l2 = ufunc(l2)
print(time() - x)