# Search: implement diag(1,1,zeta9) on qutrit x using an ancilla y prepared in basis state |c>,
# via phase gadgets: product over linear forms l of single-qutrit diag phases g_l(l(x,y)),
# each g_l = diag(1, z^a, z^b) with a+b = 0 mod 3 (Clifford if a,b = 0 mod 3, else T-type level-3).
import itertools
forms2=[(1,0),(0,1),(1,1),(1,2)]
gad=[(a,b) for a in range(9) for b in range(9) if (a+b)%3==0]
def cost(g): return 0 if (g[0]%3==0 and g[1]%3==0) else 1
def run(forms, nvars, anc_vals):
    best=None
    for gs in itertools.product(gad, repeat=len(forms)):
        c=sum(cost(g) for g in gs)
        if best is not None and c>=best[0]: continue
        ok=True; ph=[]
        for x in range(3):
            v=(x,)+anc_vals
            tot=0
            for f,g in zip(forms,gs):
                l=sum(fi*vi for fi,vi in zip(f,v))%3
                tot+= (0,g[0],g[1])[l]
            ph.append(tot%9)
        d=[(p-ph[0])%9 for p in ph]
        if d==[0,0,1]: best=(c,gs)
    return best
for c in range(3):
    print("2 qutrits, ancilla |%d>:"%c, run(forms2,2,(c,)))
