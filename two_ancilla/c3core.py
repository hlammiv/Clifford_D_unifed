"""c3core.py -- exact Clifford-equivalence for isometries M : C^3 -> C^27 (system (x) 2 ancillas).

Question: does there exist a 3-qutrit Clifford C, a 1-qutrit Clifford G and a phase with
        C M1 G = e^{i phi} M2 ?
Method.
 * labels L_M[P, A] = |Tr(A^dag M^dag P M)|^2 / 9  (P: 729 3-qutrit Paulis X^a Z^b, A: 9 1-qutrit Paulis).
   If C has symplectic map S, then L_{C M G}[S P, A] = L_{M G}[P, A]  (phases drop out).
 * Loop over the 24 symplectic classes of G (G mod Paulis); replace M1 by M1 G.
 * V = span{P : L[P,.] != 0}.  The Choi state of M is determined by its characteristic function,
   which is supported on V, so only S|_V matters (two Cliffords agreeing on V differ, on the Choi
   state, by a Pauli -- see argument in README section of the report).  Backtrack over linear
   injective isometric maps S|_V : V -> F_3^6 that preserve the label rows (images of a basis of V,
   checking every new span element).  Extend to a full symplectic S (Witt; found by DFS).
 * Build a Clifford U with symplectic part S, and test all 729 x 9 Pauli corrections Q (x) A:
   C = Q U, G' = G A^T-ish ;  |Tr(A^dag M2^dag Q U M1 G)| = 3  <=>  Q U M1 G A ~ M2 (exactly up to phase).
   The returned (C, G') is re-verified explicitly with allclose.
 Every step is exhaustive, so equiv() returning None is a proof of non-equivalence (given that the
 floating label quantisation is stable; the rounding margin is tracked in MARGIN).
"""
import numpy as np, itertools

z = np.exp(2j*np.pi/9); w = z**3
n = 3; D = 27
LABS = np.array(list(itertools.product(range(3), repeat=2*n)))      # (a1,b1,a2,b2,a3,b3), index = base-3
DIG = np.array(list(itertools.product(range(3), repeat=n)))
PW = 3**np.arange(n)[::-1]
SRC = np.zeros((729, D), np.int64); PH = np.zeros((729, D), complex)   # (P M)[y] = PH[y] * M[SRC[y]]
for k, lab in enumerate(LABS):
    a = lab[0::2]; b = lab[1::2]
    x = (DIG - a) % 3                     # P|x> = w^{b.x}|x+a>  -> (P M)[y] = w^{b.(y-a)} M[y-a]
    SRC[k] = x @ PW; PH[k] = w**((x*b).sum(1) % 3)
def pmat(k):
    M = np.zeros((D, D), complex); M[np.arange(D), SRC[k]] = PH[k]; return M
PM = np.array([pmat(k) for k in range(729)])
X1 = np.roll(np.eye(3), 1, axis=0); Z1 = np.diag([1, w, w**2])
mp = np.linalg.matrix_power
P1 = np.array([mp(X1, a) @ mp(Z1, b) for a in range(3) for b in range(3)])
P1c = np.conj(P1)
H1 = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(3)
S1 = np.diag([1, 1, w])

def vec2idx(v): return int(np.dot(v % 3, 3**np.arange(2*n)[::-1]))
IDX2VEC = LABS.copy()
# symplectic form  <u,v> = sum_i a_i b'_i - b_i a'_i  (mod 3)
def sform(u, v): return int((u[0::2]*v[1::2] - u[1::2]*v[0::2]).sum() % 3)
A_ = LABS[:, 0::2]; B_ = LABS[:, 1::2]
SFORM = ((A_ @ B_.T - B_ @ A_.T) % 3).astype(np.int8)                 # 729 x 729
ADD = ((LABS[:, None, :] + LABS[None, :, :]) % 3) @ (3**np.arange(2*n)[::-1])   # idx(u+v)
DBL = ADD[np.arange(729), np.arange(729)]                             # idx(2u)

Q = 1e6
MARGIN = [0.5]

def raw_labels(Ms):
    """Ms (N,27,3) -> (N,729,9) real |Tr(A^dag M^dag P M)|^2/9"""
    PMs = PH[None, :, :, None] * Ms[:, SRC, :]                        # N,729,27,3
    Y = np.einsum('nya,npyb->npab', np.conj(Ms), PMs, optimize=True)  # N,729,3,3
    c = np.einsum('Aab,npab->npA', P1c, Y, optimize=True)
    return (c.real**2 + c.imag**2)/9

def qlabels(Ms):
    v = raw_labels(Ms)*Q; r = np.rint(v)
    MARGIN[0] = min(MARGIN[0], float(np.min(0.5-np.abs(v-r))))
    return r.astype(np.int64)

def _mix(a, axis):
    a = np.sort(a, axis=axis)
    h = np.zeros(np.delete(a.shape, axis), dtype=np.uint64); m = np.uint64(1099511628211)
    with np.errstate(over='ignore'):
        for k in range(a.shape[axis]):
            h = (h ^ np.take(a, k, axis=axis).astype(np.uint64)) * m + np.uint64(k+7)
    return h

def inv_hash(L):
    """L (N,729,9) quantised labels -> (N,) uint64 hash invariant under C and G."""
    col = _mix(L, 2)                       # N,729 per-P multiset over A
    row = _mix(L, 1)                       # N,9   per-A multiset over P
    with np.errstate(over='ignore'):
        return _mix(col.view(np.int64), 1) ^ (_mix(row.view(np.int64), 1)*np.uint64(31))

# ---- single-qutrit Cliffords: 24 symplectic classes ----
def _cliff1():
    def key(U):
        v = U.flatten(); i = np.argmax(abs(v) > 1e-9); v = v/(v[i]/abs(v[i])); return tuple(np.round(v, 8))
    out = {key(np.eye(3)): np.eye(3, dtype=complex)}; fr = [np.eye(3, dtype=complex)]
    while fr:
        nf = []
        for U in fr:
            for g in (H1, S1, X1):
                V = g @ U; k = key(V)
                if k not in out: out[k] = V; nf.append(V)
        fr = nf
    return list(out.values())
CL1 = _cliff1(); assert len(CL1) == 216
def _p1idx(M):
    for i in range(9):
        if abs(abs(np.trace(P1[i].conj().T @ M))/3-1) < 1e-9: return i
    raise ValueError
_seen = {}
for G in CL1:
    k = tuple(_p1idx(G @ A @ G.conj().T) for A in P1)
    _seen.setdefault(k, G)
GREPS = list(_seen.values()); assert len(GREPS) == 24

# ---- Clifford unitary from a full symplectic map (images of x_j, z_j as label indices) ----
def clifford_from_images(img):
    """img: list of 6 label indices = S(x1),S(z1),S(x2),S(z2),S(x3),S(z3). Returns 27x27 unitary U with
    U X_j U^dag = P(S x_j), U Z_j U^dag ~ P(S z_j) (up to phase)."""
    Xs = [PM[img[2*j]] for j in range(3)]; Zs = [PM[img[2*j+1]] for j in range(3)]
    Pi = np.eye(D, dtype=complex)
    for Zt in Zs: Pi = Pi @ (np.eye(D) + Zt + Zt @ Zt)/3
    j = np.argmax(np.linalg.norm(Pi, axis=0)); u0 = Pi[:, j]/np.linalg.norm(Pi[:, j])
    U = np.zeros((D, D), complex)
    for xi, xd in enumerate(DIG):
        v = u0
        for q in range(3):
            for _ in range(xd[q]): v = Xs[q] @ v
        U[:, xi] = v
    return U

STD = [vec2idx(np.eye(6, dtype=int)[i]) for i in range(6)]

def _span_add(span, b, w_):
    """span: dict vec_idx -> img_idx.  returns list of new (v, img) for v = u + c b."""
    new = []
    b2 = DBL[b]; w2 = DBL[w_]
    for u, iu in span.items():
        new.append((ADD[u, b], ADD[iu, w_])); new.append((ADD[u, b2], ADD[iu, w2]))
    return new

def _extend(basis, imgs):
    """Extend isometry (basis -> imgs) to full symplectic map; return images of STD basis, or None."""
    # complete basis with standard vectors
    span = {0: 0}
    for b, wi in zip(basis, imgs):
        for v, iv in _span_add(span, b, wi): span[v] = iv
    full_b = list(basis); full_w = list(imgs)
    extra = []
    sp = set(span.keys())
    for e in STD:
        if e not in sp:
            extra.append(e); ns = set(sp)
            for u in sp: ns.add(ADD[u, e]); ns.add(ADD[u, DBL[e]])
            sp = ns
    def dfs(k, span_img, cur_b, cur_w):
        if k == len(extra): return cur_w
        e = extra[k]
        need = [SFORM[e, b] for b in cur_b]
        ok = np.all(SFORM[:, cur_w] == np.array(need, dtype=np.int8)[None, :], axis=1)
        for cand in np.nonzero(ok)[0]:
            if cand in span_img: continue
            ns = set(span_img)
            for u in span_img: ns.add(ADD[u, cand]); ns.add(ADD[u, DBL[cand]])
            r = dfs(k+1, ns, cur_b+[e], cur_w+[int(cand)])
            if r is not None: return r
        return None
    span_img = set(span.values())
    if len(span_img) != len(span): return None
    res = dfs(0, span_img, full_b, full_w)
    if res is None: return None
    allb = full_b + extra
    # images of STD basis: solve linear system mod 3. Build full map via span enumeration.
    span = {0: 0}
    for b, wi in zip(allb, res):
        for v, iv in _span_add(span, b, wi): span[v] = iv
    return [span[e] for e in STD]

def choi_overlap(X, M2):
    """|Tr(A^dag M2^dag Q X)| for all Q (729), A (9)  -> (729,9)"""
    QX = PH[:, :, None] * X[SRC, :]
    Y = np.einsum('ya,pyb->pab', np.conj(M2), QX, optimize=True)
    return np.abs(np.einsum('Aab,pab->pA', P1c, Y, optimize=True))

STATS = dict(maps=0, tests=0)

def ykeys(M, q=1e6):
    """phase-sensitive per-Pauli labels: Y_P = M^dag P M modulo multiplication by omega
    (a Clifford maps P to w^k P(SP), so Y_{CMG}[SP] = w^-k G^dag Y_M[P] G).  Returns list of 729 bytes keys."""
    PMs = PH[:, :, None] * M[SRC, :]
    Y = np.einsum('ya,pyb->pab', np.conj(M), PMs, optimize=True).reshape(729, 9)
    cands = []
    for k in range(3):
        Yk = Y*w**k
        v = np.concatenate([Yk.real, Yk.imag], axis=1)*q
        r = np.rint(v); MARGIN[0] = min(MARGIN[0], float(np.min(0.5-np.abs(v-r))))
        cands.append(r.astype(np.int64) + 0)
    C = np.stack(cands, 1)                                   # 729,3,18
    return [min(C[p, 0].tobytes(), C[p, 1].tobytes(), C[p, 2].tobytes()) for p in range(729)]

ZEROKEY = None

def equiv(M1, M2, K2=None):
    """Exact test. Returns (C, G, phase) with C @ M1 @ G == phase * M2, or None.
    Loops over all 216 one-qutrit Cliffords G (mod phase); for each, backtracks over linear injective
    symplectic-form-preserving maps S on V = span{P : Y_{M1G}[P] != 0} with Y_{M1 G}[P] ~ Y_{M2}[S P]
    (equal modulo a power of omega: a necessary condition), extends S to Sp(6,3), builds a Clifford U
    for it and finishes with an exhaustive Pauli-correction test + explicit matrix check."""
    global ZEROKEY
    STATS['tests'] += 1
    if K2 is None: K2 = ykeys(M2)
    if ZEROKEY is None: ZEROKEY = ykeys(np.zeros((D, 3)))[0]
    for G in CL1:
        M1g = M1 @ G
        K1 = ykeys(M1g)
        ids = {}
        r1 = np.array([ids.setdefault(k, len(ids)) for k in K1]); r2 = np.array([ids.setdefault(k, len(ids)) for k in K2])
        if not np.array_equal(np.sort(r1), np.sort(r2)): continue
        zid = ids.get(ZEROKEY, -1)
        cnt = np.bincount(r2, minlength=len(ids))
        nz = [i for i in range(1, 729) if r1[i] != zid]
        nz.sort(key=lambda i: cnt[r1[i]])
        basis = []; sp = {0}
        for i in nz:
            if i in sp: continue
            basis.append(i); ns = set(sp)
            for u in sp: ns.add(ADD[u, i]); ns.add(ADD[u, DBL[i]])
            sp = ns
        cands = [np.nonzero(r2 == r1[b])[0] for b in basis]
        def bt(k, span, imgs, imgset):
            if k == len(basis):
                STATS['maps'] += 1
                full = _extend(basis, imgs)
                if full is None: return None
                U = clifford_from_images(full)
                X = U @ M1g
                co = choi_overlap(X, M2)
                hit = np.argwhere(co > 3-1e-7)
                if len(hit):
                    q, a = hit[0]
                    C = PM[q] @ U
                    for Gt in (G @ P1[a].conj().T, G @ P1[a].T, G @ np.conj(P1[a]), G @ P1[a]):
                        Y = C @ M1 @ Gt
                        ph = np.vdot(M2, Y)/3
                        if abs(abs(ph)-1) < 1e-8 and np.allclose(Y, ph*M2, atol=1e-8):
                            return (C, Gt, ph)
                    raise RuntimeError("overlap hit but explicit verification failed")
                return None
            b = basis[k]
            for wc in cands[k]:
                wc = int(wc)
                if wc in imgset: continue
                if any(SFORM[wc, imgs[i]] != SFORM[b, basis[i]] for i in range(k)): continue
                new = _span_add(span, b, wc)
                ok = True
                for v, iv in new:
                    if r2[iv] != r1[v] or iv in imgset: ok = False; break
                if not ok: continue
                newimgs = [iv for _, iv in new]
                if len(set(newimgs)) != len(newimgs): continue
                ns = dict(span); ns.update(new)
                r = bt(k+1, ns, imgs+[wc], imgset | set(newimgs))
                if r is not None: return r
            return None
        r = bt(0, {0: 0}, [], {0})
        if r is not None: return r
    return None

# ---- T-rotations f(P) for the 728 non-identity Paulis; f(P)^{-1} = f(P^2) ----
def fP(P):
    fv = [1, z, z**8]; out = np.zeros((D, D), complex); Pk = [np.eye(D), P, P @ P]
    for m in range(3): out += fv[m]*sum(w**(-m*k)*Pk[k] for k in range(3))/3
    return out
ROT = np.array([fP(PM[k]) for k in range(1, 729)])     # ROT[k-1] = f(P_k)
E00 = np.zeros((D, 3), complex)
for x in range(3): E00[9*x, x] = 1

def gate1(U, q):
    ops = [np.eye(3)]*3; ops = list(ops); ops[q] = U
    return np.kron(np.kron(ops[0], ops[1]), ops[2])
def sum_gate(c, t):
    M = np.zeros((D, D))
    for xi, xd in enumerate(DIG):
        y = xd.copy(); y[t] = (y[t]+y[c]) % 3; M[y @ PW, xi] = 1
    return M
CLGENS = [gate1(H1, q) for q in range(3)] + [gate1(S1, q) for q in range(3)] + \
         [sum_gate(c, t) for c in range(3) for t in range(3) if c != t]
def random_clifford(rng, L=60):
    U = np.eye(D, dtype=complex)
    for _ in range(L): U = CLGENS[rng.integers(len(CLGENS))] @ U
    return U
