import itertools, numpy as np
exec(open('mincount.py').read().split('for c in range(9)')[0])
# T-type restricted: allow lifts, gate class (0,a,b) with lift; T-type iff exponents sum = 0 mod 3 -> for class (0,a,b) sum a+b mod3 must be 0
def best_T(f):
  res=None
  for choice in itertools.product(classes,repeat=4):
    if any((g[1]+g[2])%3!=0 for g in choice): continue
    tot=[0]*9
    for (a,b),g in zip(lines,choice):
      for p,(x,y) in enumerate(pts): tot[p]+=g[(a*x+b*y)%3]
    r=[(f[p]-tot[p])%9 for p in range(9)]; r=[(v-r[0])%9 for v in r]
    if any(v%3 for v in r): continue
    if tuple((v//3)%3 for v in r) in Q:
      n=sum(1 for g in choice if g!=(0,0,0))
      if res is None or n<res[0]: res=(n,choice)
  return res
for c in range(9):
  f=[(c*(x==2)+3*((x==2) and (y!=0)))%9 for (x,y) in pts]
  print('T-type only c=',c,best_T(f))
# numeric verification of 3-D-gate C2(zeta^7 Z(1,1)) (exact phase table incl. Clifford residual)
z=np.exp(2j*np.pi/9); w=z**3
choice=((0,0,0),(0,1,1),(0,0,2),(0,0,2))
f=[(7*(x==2)+3*((x==2) and (y!=0)))%9 for (x,y) in pts]
tot=[sum(g[(a*x+b*y)%3] for (a,b),g in zip(lines,choice)) for (x,y) in pts]
r=[(f[p]-tot[p])%9 for p in range(9)]
print('residual (should be const + 3*quadratic):',r)
