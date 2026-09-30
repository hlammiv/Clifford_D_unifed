# Can f(x,y) (Z_9-valued phase exponent of zeta) be written as sum over 4 directions of g_l(a x + b y mod 3), modulo 9,
# up to Clifford-diagonal freedom 3*(quadratic in x,y) + const ?
import itertools, numpy as np
pts=[(x,y) for x in range(3) for y in range(3)]
lines=[(1,0),(0,1),(1,1),(1,2)]
# generator columns: for each line l and value v in Z3, indicator [a x + b y == v]
cols=[]; labels=[]
for (a,b) in lines:
  for v in range(3):
    cols.append([1 if (a*x+b*y)%3==v else 0 for (x,y) in pts]); labels.append(((a,b),v))
# Clifford freedom: 3*monomials of degree<=2: 1,x,y,x^2,xy,y^2
for m in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]:
  cols.append([3*(x**m[0])*(y**m[1]) for (x,y) in pts]); labels.append(('cliff',m))
M=np.array(cols).T  # 9 x ncols
def target(x,y): return (7*(x==2) + 3*((x==2) and (y!=0)))%9
f=np.array([target(x,y) for (x,y) in pts])
# brute force: search coefficients mod 9 is 9^(18) too big; do Smith-normal-form style via sympy over Z then mod 9
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form
# solve M c = f + 9 k : augment with 9*I
A=Matrix(np.hstack([M, 9*np.eye(9,dtype=int)]).tolist())
# use hermite-type solving via lattice: check f in column lattice of A
from sympy import symbols
import sympy
# Simple approach: integer lattice membership using HNF
from sympy.matrices.normalforms import hermite_normal_form
H=hermite_normal_form(A)  # columns basis
print(H.shape)
sol=H.solve(Matrix(f.tolist())) if H.shape[0]==H.shape[1] else None
print(sol.T if sol is not None else None, all(s.is_integer for s in sol) if sol is not None else None)
