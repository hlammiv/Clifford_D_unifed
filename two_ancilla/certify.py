"""certify.py -- re-examine the fwd/bwd invariant collisions from orbits3.py (depth 3) and give an
independent, simple non-equivalence certificate: for every one of the 216 one-qutrit Cliffords G,
the multiset {Y_P modulo omega : P in 729 Paulis}, Y_P = (M_f G)^dag P (M_f G), differs from that of M_b.
(Necessary condition for C M_f G ~ M_b: a Clifford maps P -> w^k P(SP), so the multisets coincide.)"""
import numpy as np, pickle, sys, collections
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
import c3core as cc
fw = pickle.load(open('out/fwd_3.pkl', 'rb'))
for name in ('L4k1', 'R'):
    bw = pickle.load(open(f'out/bwd_{name}_3.pkl', 'rb'))
    fidx = collections.defaultdict(list)
    for k, h in enumerate(fw['h']): fidx[h].append(k)
    for j, h in enumerate(bw['h']):
        for k in fidx.get(h, []):
            Mf, Mb = fw['M'][k], bw['M'][j]
            Kb = collections.Counter(cc.ykeys(Mb))
            nmatch = sum(collections.Counter(cc.ykeys(Mf @ G)) == Kb for G in cc.CL1)
            # magnitude-only multiset (the old l4_m2_lb-style invariant) for comparison
            Lf = np.sort(cc.qlabels(Mf[None])[0].reshape(-1)); Lb = np.sort(cc.qlabels(Mb[None])[0].reshape(-1))
            r = cc.equiv(Mf, Mb)
            print(f"{name}: collision fwd orbit {k} (depth {fw['depth'][k]}, seq {fw['seq'][k]}) vs bwd orbit {j} "
                  f"(depth {bw['depth'][j]}, seq {bw['seq'][j]}); |.|^2 label multisets equal: {np.array_equal(Lf, Lb)}; "
                  f"#G with equal omega-class multiset of Y_P: {nmatch}/216; exact equiv: {r is not None}")
print("margin", cc.MARGIN[0])
