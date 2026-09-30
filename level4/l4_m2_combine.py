"""combine m2parts/*.npy : min t over fwd depth d1 + bwd depth d2 with an invariant match."""
import numpy as np, glob, sys
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
def load(side):
    ds = [set(), set(), set(), set()]
    fs = sorted(glob.glob(f"/home/hlamm/Desktop/efficent_gates/unified/level4/m2parts/m2_{side}_{k if side=='bwd' else 1}_*.npy"))
    for f in fs:
        a = np.load(f, allow_pickle=True)
        for d in range(4): ds[d] |= a[d]
    return ds, len(fs)
F, nf = load("fwd"); B, nb = load("bwd")
print("files", nf, nb, "distinct invariants fwd", [len(x) for x in F], "bwd", [len(x) for x in B])
best = None
for d1 in range(4):
    for d2 in range(4):
        if F[d1] & B[d2]:
            print(f"  match fwd depth {d1} + bwd depth {d2} = t {d1+d2}")
            if best is None or d1+d2 < best: best = d1+d2
print("min matching t (<= 6 searched):", best)
