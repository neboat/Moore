import sys, random
from time import *

n = 4096                                 ##\lilabel{set_n}

A = [[random.random()
      for row in xrange(n)]
     for col in xrange(n)] ##\lilabel{rand(}
B = [[random.random()
      for row in xrange(n)]
     for col in xrange(n)] ##\lilabel{rand)}
C = [[0 for row in xrange(n)]
     for col in xrange(n)] ##\lilabel{zero_C}

start = time()                           

for i in xrange(n):                      ##\lilabel{loop_i}\lilabel{loops(}
    for j in xrange(n):                  ##\lilabel{loop_j}
        for k in xrange(n):              ##\lilabel{loop_k}
            C[i][j] += A[i][k] * B[k][j] ##\lilabel{multiply}\lilabel{loops)}

end = time()

print '%0.6f' % (end - start)
