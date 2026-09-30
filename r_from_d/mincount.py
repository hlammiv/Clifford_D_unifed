import itertools, numpy as np
pts=[(x,y) for x in range(3) for y in range(3)]
lines=[(1,0),(0,1),(1,1),(1,2)]
# quadratic space mod 3
quad=[[(x**i)*(y**j)%3 for (x,y) in pts] for (i,j) in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]]
Q=set()
for c in itertools.product(range(3),repeat=6):
  Q.add(tuple(sum(c[k]*quad[k][p] for k in range(6))%3 for p in range(9)))
classes=[(0,a,b) for a in range(3) for b in range(3)]  # g values mod 3 with g(0)=0
def best(f):
  res=None
  for choice in itertools.product(classes,repeat=4):
    tot=[0]*9
    for (a,b),g in zip(lines,choice):
      for p,(x,y) in enumerate(pts): tot[p]+=g[(a*x+b*y)%3]
    r=[(f[p]-tot[p])%9 for p in range(9)]
    r=[(v-r[0])%9 for v in r]
    if any(v%3 for v in r): continue
    if tuple((v//3)%3 for v in r) in Q:
      n=sum(1 for g in choice if g!=(0,0,0))
      if res is None or n<res[0]: res=(n,choice)
  return res
for c in range(9):
  f=[(c*(x==2)+3*((x==2) and (y!=0)))%9 for (x,y) in pts]
  print('c=',c,'C2(Z(1,1)) with ctrl phase zeta^c:',best(f))
# also: |2>-controlled -1 would need 18th roots; check C2(S^dagger)-type: 
