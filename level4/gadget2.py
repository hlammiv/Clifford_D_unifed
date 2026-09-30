import itertools
forms=[(1,0),(0,1),(1,1),(1,2)]
gad=[(a,b) for a in range(9) for b in range(9)]
def kind(g):
    a,b=g
    if a%3==0 and b%3==0: return 'C'
    if (a+b)%3==0: return 'T'
    return 'L4'
target={(x,y):(3*(x==2)*(y==2))%9 for x in range(3) for y in range(3)}
# unique decomposition up to constants: solve by brute force over gadgets for forms 0..3
sols=[]
for gs in itertools.product(gad,repeat=4):
    ok=True; ref=None
    for x in range(3):
        for y in range(3):
            tot=sum((0,g[0],g[1])[(f[0]*x+f[1]*y)%3] for f,g in zip(forms,gs))
            d=(tot-target[(x,y)])%9
            if ref is None: ref=d
            elif d!=ref: ok=False;break
        if not ok: break
    if ok: sols.append([kind(g) for g in gs])
from collections import Counter
print(len(sols), Counter(tuple(sorted(s)) for s in sols))
