import numpy as np
import subroutines as sr

# define fake g
g = np.random.randn(3,8,8)

Ufake = sr.U(g)

print Ufake
print ''
print g

print ''
print 'Now testing max\n'
print Ufake[1]
print ''
for row in g[1]:
    print g[0,0] + row