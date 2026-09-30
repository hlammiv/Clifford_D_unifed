"""l4_m2_lb.py -- merge-free invariant MITM lower bound for diag(1,1,z^k) with TWO clean ancillas.
3 qutrits (sys, a1, a2), 728 T-rotations R(P).  J = R_s..R_1 E0 (27x3).  No orbit merging at all:
every rotation sequence is enumerated (only exact algebraic identities are pruned: R(P)R(P)=R(P^2)C,
R(P)R(P^2)=1, commuting neighbours sorted).  First step is reduced to representatives of Pauli
orbits under Stab(E0) (fwd: {T on sys, T on X-of-ancilla}; bwd: 8 system Paulis + X-of-ancilla),
which is exact (see report).  A match of a Clifford-invariant between the fwd and bwd sets is
NECESSARY for a solution, so 'no match for all s+s'<=t' proves T-count > t with 2 ancillas.
Invariant: multiset over the 729 Paulis P of (|Tr M_P|^2, ||M_P||_F^2, |det M_P|^2), M_P = J^dag P J.
usage: python3 l4_m2_lb.py DF DB"""
import numpy as np, sys, time, itertools
z = np.exp(2j*np.pi/9); w = z**3
n = 3; D = 27
labs = list(itertools.product(range(3), repeat=2*n))           # (a1,b1,a2,b2,a3,b3)
idx = np.array(list(itertools.product(range(3), repeat=n)))     # basis r -> digits
def pauli_perm_phase(lab):
    a = np.array(lab[0::2]); b = np.array(lab[1::2])
    dst = [int(np.ravel_multi_index(tuple((r + a) % 3), (3,)*n)) for r in idx]
    ph = np.array([w**int(np.dot(b, r) % 3) for r in idx])
    return np.array(dst), ph
PP = [pauli_perm_phase(l) for l in labs]
def pauli_mat(k):
    dst, ph = PP[k]; M = np.zeros((D, D), complex); M[dst, np.arange(D)] = ph; return M
def fP(P):
    fv = [1, z, z**8]; out = np.zeros((D, D), complex)
    for m in range(3): out += fv[m]*sum(w**(-m*k)*np.linalg.matrix_power(P, k) for k in range(3))/3
    return out
NR = len(labs) - 1                                              # 728
ROT = np.array([fP(pauli_mat(k)) for k in range(1, len(labs))])
ROTD = np.conj(np.transpose(ROT, (0, 2, 1)))
def symp(l1, l2): return sum(l1[2*i]*l2[2*i+1] - l1[2*i+1]*l2[2*i] for i in range(n)) % 3
L2 = [tuple((2*x) % 3 for x in l) for l in labs]
LINE = np.array([min(k, labs.index(L2[k])) for k in range(1, len(labs))])
COMM = np.array([[symp(labs[i], labs[j]) == 0 for j in range(1, len(labs))] for i in range(1, len(labs))])
ALLOW = np.ones((NR, NR), bool)
for p in range(NR):
    ALLOW[p, LINE == LINE[p]] = False
    ALLOW[p, np.nonzero(COMM[p])[0][np.nonzero(COMM[p])[0] < p]] = False
def rot_index(lab): return labs.index(tuple(lab)) - 1
E0 = np.zeros((D, 3), complex)
for x in range(3): E0[9*x, x] = 1
# gather tables for all 729 Paulis
DST = np.array([PP[k][0] for k in range(len(labs))]); PH = np.array([PP[k][1] for k in range(len(labs))])
INV_DST = np.argsort(DST, axis=1)
X1 = np.roll(np.eye(3), 1, axis=0); Z1 = np.diag([1, w, w**2])
A1c = np.conj(np.array([np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, b) for a in range(3) for b in range(3)]))
MARG = [1.0]
SHIFT = np.array([[int(np.ravel_multi_index(tuple((r + a) % 3), (3,)*n)) for r in idx] for a in idx])   # [a, r] -> r+a
def invariant(J):
    """J: (N,27,3) -> (N,) uint64 hash."""
    N = len(J)
    # M_P[al,be] = sum_r conj(J[r+a,al]) w^{b.r} J[r,be]  (all 27 shifts a, all 27 b via 3-D FFT)
    G = np.conj(J[:, SHIFT, :])[:, :, :, :, None] * J[:, None, :, None, :]                # N,27a,27r,3,3
    G = G.reshape(N, 27, 3, 3, 3, 3, 3)
    M = (np.fft.ifftn(G, axes=(2, 3, 4)) * 27).reshape(N, 729, 3, 3)
    c = np.einsum('Aab,npab->npA', A1c, M, optimize=True)                                 # Tr(A^dag M)
    v = np.rint((c.real**2 + c.imag**2)*(1e5/9)).astype(np.int64)                       # N,729,9
    MARG[0] = min(MARG[0], float(np.min(0.5-np.abs((c.real**2 + c.imag**2)*(1e5/9) - v))))
    vs = np.sort(v, axis=2)                                                               # per P over A
    colh = np.zeros(vs.shape[:2], np.uint64); m0 = np.uint64(1469598103934665603)
    with np.errstate(over='ignore'):
        for a in range(9): colh = (colh ^ vs[:, :, a].astype(np.uint64)) * m0 + np.uint64(a+3)
        rows = np.sort(v, axis=1)                                                         # per A over P
        rowh = np.zeros((N, 9), np.uint64)
        for p in range(0, rows.shape[1], 1): rowh = (rowh ^ rows[:, p, :].astype(np.uint64)) * m0 + np.uint64(p+5)
    f = np.concatenate([np.sort(colh, axis=1), np.sort(rowh, axis=1)], axis=1)
    h = np.zeros(N, np.uint64); m = np.uint64(1099511628211)
    with np.errstate(over='ignore'):
        for k in range(f.shape[1]): h = (h ^ f[:, k]) * m + np.uint64(k+1)
    return h
def enum(J0s, first, rot, depth, chunk=1500):
    """yield (depth, hashes) for all sequences; J0s: start states; first: allowed first rotations."""
    yield 0, invariant(J0s)
    lvl = []   # list of (J, last)
    J1 = np.concatenate([np.einsum('ij,njk->nik', rot[p], J0s) for p in first])
    last1 = np.repeat(np.array(first), len(J0s))
    yield 1, invariant(J1)
    if depth < 2: return
    def expand(J, last):
        for p in range(NR):
            sel = np.nonzero(ALLOW[last, p])[0]
            if len(sel): yield np.einsum('ij,njk->nik', rot[p], J[sel]), np.full(len(sel), p)
    if depth == 2:
        for J2, l2 in expand(J1, last1): yield 2, invariant(J2)
        return
    J2s, L2s = [], []
    for J2, l2 in expand(J1, last1): J2s.append(J2); L2s.append(l2)
    J2 = np.concatenate(J2s); l2 = np.concatenate(L2s); del J2s, L2s
    for c in range(0, len(J2), 2000):
        yield 2, invariant(J2[c:c+2000])
    for c in range(0, len(J2), 40):
        for J3, l3 in expand(J2[c:c+40], l2[c:c+40]):
            for cc in range(0, len(J3), chunk): yield 3, invariant(J3[cc:cc+chunk])
def depth_sets(side, k):
    if side == "fwd":
        J0 = E0; first = [rot_index((0, 1, 0, 0, 0, 0)), rot_index((0, 0, 1, 0, 0, 0))]; rot = ROT
    else:
        J0 = E0 @ np.diag([1, 1, z**k]); rot = ROTD
        first = [rot_index((a, b, 0, 0, 0, 0)) for a in range(3) for b in range(3) if (a, b) != (0, 0)] + [rot_index((0, 0, 1, 0, 0, 0))]
    return J0[None], first, rot
def part_main():
    side, k, NP, I = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    J0s, first, rot = depth_sets(side, k)
    hs = {0: set(), 1: set(), 2: set(), 3: set()}
    J1 = np.concatenate([np.einsum('ij,njk->nik', rot[p], J0s) for p in first]); l1 = np.repeat(np.array(first), len(J0s))
    J2s, L2s = [], []
    for p in range(NR):
        sel = np.nonzero(ALLOW[l1, p])[0]
        if len(sel): J2s.append(np.einsum('ij,njk->nik', rot[p], J1[sel])); L2s.append(np.full(len(sel), p))
    J2 = np.concatenate(J2s); l2 = np.concatenate(L2s)
    if I == 0:
        hs[0] |= set(invariant(J0s).tolist()); hs[1] |= set(invariant(J1).tolist())
        for c in range(0, len(J2), 1000): hs[2] |= set(invariant(J2[c:c+1000]).tolist())
    mine = np.arange(I, len(J2), NP); t0 = time.time(); ns = 0
    for c in range(0, len(mine), 20):
        sl = mine[c:c+20]; Jc = J2[sl]; lc = l2[sl]
        buf = []
        for p in range(NR):
            sel = np.nonzero(ALLOW[lc, p])[0]
            if len(sel): buf.append(np.einsum('ij,njk->nik', rot[p], Jc[sel]))
        J3 = np.concatenate(buf); ns += len(J3)
        for cc in range(0, len(J3), 1500): hs[3] |= set(invariant(J3[cc:cc+1500]).tolist())
    np.save(f"m2parts/m2_{side}_{k}_{I}.npy", np.array([hs[d] for d in range(4)], dtype=object), allow_pickle=True)
    print(f"part {side} k={k} {I}/{NP}: {ns} depth-3 states, margin {MARG[0]:.4f}, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__" and sys.argv[1] == "part":
    part_main()
elif __name__ == "__main__":
    DF, DB = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    fwd_first = [rot_index((0, 1, 0, 0, 0, 0)), rot_index((0, 0, 1, 0, 0, 0))]   # T on sys ; f(X_a1)
    fwd = {}
    cnt = [0]*4
    for d, h in enum(E0[None], fwd_first, ROT, DF):
        cnt[d] += len(h)
        for x in np.unique(h).tolist(): fwd.setdefault(x, d)
    print(f"fwd states per depth {cnt}, {len(fwd)} distinct invariants [{time.time()-t0:.0f}s]", flush=True)
    for k in (1, 2):
        L = np.diag([1, 1, z**k]); JT = E0 @ L
        bwd_first = [rot_index((a, b, 0, 0, 0, 0)) for a in range(3) for b in range(3) if (a, b) != (0, 0)] + [rot_index((0, 0, 1, 0, 0, 0))]
        best = None; cnt = [0]*4
        for d, h in enum(JT[None], bwd_first, ROTD, DB):
            cnt[d] += len(h)
            for x in np.unique(h).tolist():
                if x in fwd:
                    t = fwd[x] + d
                    if best is None or t < best: best = t; print(f"  k={k}: invariant match at t={t} (fwd {fwd[x]}, bwd {d})", flush=True)
        print("   min rounding margin so far:", MARG[0])
        print(f"k={k}: bwd states per depth {cnt}; min invariant-match t = {best}  [{time.time()-t0:.0f}s]", flush=True)

