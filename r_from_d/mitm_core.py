"""mitm_core.py -- exact T-count search for 2-qutrit Clifford+T isometries (system x 1 clean ancilla).

Every Clifford+T circuit is W = C * R_{P_t} ... R_{P_1}, with R_P = f(P) = sum_j zeta^{t(j)} Pi_j(P)
(t = (0,1,8)), i.e. a Clifford conjugate of T, for P one of the 80 non-identity 2-qutrit Paulis.
R_P^{-1} = R_{P^2}.  The T-count of W is t.

Isometries M = W (I (x) |0>) (9x3) are classified modulo LEFT Clifford multiplication and global phase
by the invariant  Inv(M) = multiset_P { M^dag P M  modulo omega-phase }  (P over 80 Paulis).
M^dag P M is in Z[zeta][1/3]; it is computed in the three Galois embeddings zeta->zeta^a (a=1,2,4),
converted to EXACT integer coordinates in the basis 1,zeta,...,zeta^5 (times 9^SCALE_D), the omega-phase
is fixed canonically, and the multiset is hashed (order-independent sum of 64-bit mixes).

Target: exists Clifford C with  A B E0 = C R E0  <=>  class(B E0) = class(A' R E0), A' in products of
inverse rotations (= rotations, as R_P^{-1}=R_{P^2}).  So we BFS from E0 and from R E0 with the same
80 generators and intersect hash sets:  T-count <= a+b  iff  level_a(E0) meets level_b(RE0).
"""
import numpy as np, numba as nb, itertools

EMB = (1, 2, 4)
SCALE_D = 7            # coordinates are exact integers of K * 9^SCALE_D  (valid for depth <= SCALE_D)

def zeta(a): return np.exp(2j*np.pi*a/9)

def paulis2():
    """80 non-identity 2-qutrit Paulis X^a Z^b (x) X^c Z^d as monomials (src,phase exponent of omega)."""
    out = []
    for a1, b1, a2, b2 in itertools.product(range(3), repeat=4):
        if (a1, b1, a2, b2) == (0, 0, 0, 0): continue
        src = np.zeros(9, np.int64); ph = np.zeros(9, np.int64)
        for y in range(9):
            y1, y2 = divmod(y, 3); x1, x2 = (y1-a1) % 3, (y2-a2) % 3
            src[y] = 3*x1+x2; ph[y] = (b1*x1+b2*x2) % 3
        out.append((src, ph))
    return out

PAULIS = paulis2()

def paulisn(n):
    out = []
    for ab in itertools.product(range(3), repeat=2*n):
        if not any(ab): continue
        a = ab[0::2]; b = ab[1::2]; D = 3**n
        src = np.zeros(D, np.int64); ph = np.zeros(D, np.int64)
        for y in range(D):
            ys = np.unravel_index(y, (3,)*n); xs = [(ys[k]-a[k]) % 3 for k in range(n)]
            src[y] = np.ravel_multi_index(xs, (3,)*n); ph[y] = sum(b[k]*xs[k] for k in range(n)) % 3
        out.append((src, ph))
    return out

def pauli_mat(src, ph, a):
    D = len(src); w = zeta(3*a); P = np.zeros((D, D), complex)
    for y in range(D): P[y, src[y]] = w**ph[y]
    return P

def rot_mat(src, ph, a):
    """R_P in embedding a: sum_k c_k P^k, c_k = (1/3) sum_j zeta^{t(j)} omega^{-jk}."""
    z = zeta(a); w = z**3; t = (0, 1, 8)
    D = len(src); P = pauli_mat(src, ph, a); out = np.zeros((D, D), complex); Pk = np.eye(D)
    for k in range(3):
        ck = sum(z**t[j]*w**(-j*k) for j in range(3))/3
        out += ck*Pk; Pk = Pk@P
    return out

ROT = np.array([[rot_mat(s, p, a) for a in EMB] for (s, p) in PAULIS])          # (80,3,9,9)
PSRC = np.array([s for s, p in PAULIS]); PPH = np.array([p for s, p in PAULIS])
OMEGA_E = np.array([zeta(3*a) for a in EMB])                                     # sigma_a(omega)
# coordinate recovery matrix
B = np.zeros((6, 6))
for i, a in enumerate(EMB):
    for j in range(6):
        v = zeta(a*j); B[2*i, j] = v.real; B[2*i+1, j] = v.imag
BINV = np.linalg.inv(B)
rng = np.random.default_rng(12345)
HCOEF = rng.integers(1, 2**62, size=(9, 6), dtype=np.int64) | 1

def e0(a=None):
    E = np.zeros((9, 3), complex)
    for x in range(3): E[3*x, x] = 1
    return E

def embed_all(Mfun):
    return np.array([Mfun(a) for a in EMB])

@nb.njit(cache=True)
def _mix(h):
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return h ^ (h >> np.uint64(31))

@nb.njit(cache=True)
def _hash_one(M, psrc, pph, omg, binv, scale, hcoef):
    # M: (3,9,3) complex, returns uint64 hash and max rounding error
    total = np.uint64(0); maxerr = 0.0
    K = np.zeros((3, 9), np.complex128)
    coords = np.zeros((9, 6), np.int64); cand = np.zeros((3, 6), np.int64)
    v = np.zeros(6)
    for p in range(psrc.shape[0]):
        for e in range(3):
            for i in range(3):
                for j in range(3):
                    s = 0j
                    for y in range(M.shape[1]):
                        s += np.conj(M[e, y, i]) * omg[e]**pph[p, y] * M[e, psrc[p, y], j]
                    K[e, 3*i+j] = s
        # find first nonzero entry and canonical omega shift
        first = -1
        for q in range(9):
            if abs(K[0, q]) > 1e-7: first = q; break
        best_s = 0
        if first >= 0:
            for s_ in range(3):
                for e in range(3):
                    val = K[e, first] * omg[e]**s_ * scale
                    v[2*e] = val.real; v[2*e+1] = val.imag
                for c in range(6):
                    x = 0.0
                    for d in range(6): x += binv[c, d]*v[d]
                    cand[s_, c] = np.int64(np.round(x))
            # lexicographic min
            best_s = 0
            for s_ in range(1, 3):
                for c in range(6):
                    if cand[s_, c] < cand[best_s, c]: best_s = s_; break
                    if cand[s_, c] > cand[best_s, c]: break
        h = np.uint64(0x9E3779B97F4A7C15)
        for q in range(9):
            for e in range(3):
                val = K[e, q] * omg[e]**best_s * scale
                v[2*e] = val.real; v[2*e+1] = val.imag
            for c in range(6):
                x = 0.0
                for d in range(6): x += binv[c, d]*v[d]
                r = np.round(x); er = abs(x-r)
                if er > maxerr: maxerr = er
                h += np.uint64(np.int64(r)) * np.uint64(hcoef[q, c])
        total += _mix(h)
    return total, maxerr

@nb.njit(parallel=True, cache=True)
def hash_batch(Ms, psrc, pph, omg, binv, scale, hcoef):
    n = Ms.shape[0]; out = np.zeros(n, np.uint64); err = np.zeros(n)
    for i in nb.prange(n):
        out[i], err[i] = _hash_one(Ms[i], psrc, pph, omg, binv, scale, hcoef)
    return out, err

@nb.njit(parallel=True, cache=True)
def children_hash(Ms, rot, psrc, pph, omg, binv, scale, hcoef):
    """hash of R_P M for every M and every of the 80 rotations -> (n,80)"""
    n = Ms.shape[0]; nr = rot.shape[0]
    out = np.zeros((n, nr), np.uint64); err = np.zeros(n)
    for i in nb.prange(n):
        dim = Ms.shape[2]
        C = np.zeros((3, dim, 3), np.complex128); me = 0.0
        for r in range(nr):
            for e in range(3):
                for y in range(dim):
                    for j in range(3):
                        s = 0j
                        for x in range(dim): s += rot[r, e, y, x]*Ms[i, e, x, j]
                        C[e, y, j] = s
            h, er = _hash_one(C, psrc, pph, omg, binv, scale, hcoef)
            out[i, r] = h
            if er > me: me = er
        err[i] = me
    return out, err

def apply_rot(Ms, r):
    return np.einsum('eyx,nexj->neyj', ROT[r], Ms)

def H(Ms):
    return hash_batch(Ms, PSRC, PPH, OMEGA_E, BINV, float(9**SCALE_D), HCOEF)

def CH(Ms):
    return children_hash(Ms, ROT, PSRC, PPH, OMEGA_E, BINV, float(9**SCALE_D), HCOEF)

def bfs(M0, depth_store, verbose=True, tag='', stream_last=True):
    """BFS from M0 (3,9,3) modulo left Clifford.  Elements stored up to depth_store; if stream_last,
    level depth_store+1 is hashed (with parent pointers) but its elements are not stored.
    Returns list of dicts per level: {'h': hashes, 'par': parent index in previous level, 'rot': rotation}
    and list of stored element arrays."""
    import time
    h0, e = H(M0[None])
    levels = [dict(h=h0, par=np.array([-1]), rot=np.array([-1]))]; Ms = [M0[None]]
    seen = h0.copy()
    for d in range(1, depth_store + (2 if stream_last else 1)):
        t0 = time.time(); prev = Ms[-1]
        hs, err = CH(prev)
        assert err.max() < 1e-3, err.max()
        flat = hs.ravel()
        uniq, idx = np.unique(flat, return_index=True)
        new = ~np.isin(uniq, seen)
        uniq = uniq[new]; idx = idx[new]
        par = idx // 80; rr = idx % 80
        levels.append(dict(h=uniq, par=par, rot=rr)); seen = np.union1d(seen, uniq)
        if verbose: print(f'  [{tag}] level {d}: {len(uniq)} new classes ({len(flat)} children, {time.time()-t0:.1f}s, maxerr {err.max():.1e})', flush=True)
        if d <= depth_store:
            newM = np.empty((len(idx), 3, 9, 3), complex)
            for r in range(80):
                sel = rr == r
                if sel.any(): newM[sel] = apply_rot(prev[par[sel]], r)
            Ms.append(newM)
    return levels, Ms

def path(levels, lev, h):
    """rotation sequence (in application order) reaching hash h at level lev"""
    i = int(np.searchsorted(levels[lev]['h'], h)); assert levels[lev]['h'][i] == h
    seq = []
    for L in range(lev, 0, -1):
        seq.append(int(levels[L]['rot'][i])); i = int(levels[L]['par'][i])
    return seq[::-1]

def meet(la, lb, tmax=99):
    """all (t,a,b,hash) minimal-t meets"""
    best = None; hits = []
    for a in range(len(la)):
        for b in range(len(lb)):
            if a+b > tmax: continue
            common = np.intersect1d(la[a]['h'], lb[b]['h'])
            if common.size:
                if best is None or a+b < best: best = a+b; hits = []
                if a+b == best: hits += [(a, b, int(c)) for c in common[:5]]
    return best, hits
