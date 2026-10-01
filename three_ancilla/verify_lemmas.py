"""verify_lemmas.py -- numerical checks of the lemmas behind the "no <=6-T circuit with any number of
clean ancillas" argument for R = diag(1,1,-1) and L4 = diag(1,1,zeta9).  Standalone (numpy only).

Setting: 4 qutrits q0 = system, q1..q3 = ancillas; T = diag(1,z,z^8); f(P) = sum_k f_k Pi_k(P)
(the Clifford conjugate of T with C Z C^dag = P);  E0 = I_sys (x) |000>.

 (1) single-qutrit magic states m_c = f(w^c X)|0>, mbar_c = f(w^c X)^{-1}|0>: non-stabilizer
     (max_{P != I} |<P>| < 1) and all have the same mana.
 (2) FRESH-STEP LEMMA, checked exhaustively over all 4374 Paulis P = X^a Z^b with a_3 != 0 and for several
     parent isometries Jt (27x3, incl. non-Clifford ones):  an explicit Clifford E built from
     M2_3, SUM(3->j), CZ(3,j), S_3 satisfies  E (Jt (x) |0>) = Jt (x) |0>  and  E P E^dag = w^c X_3,
     hence  f(P)(Jt (x) |0>) = E^dag (Jt (x) m_c).
     NON-FRESH: for all 2187 P with a_3 = 0:  f(P)(Jt (x) |0>) = (f(P~) Jt) (x) |0>,  P~ = P with Z_3 dropped.
 (3) mana of Choi states: M(Choi R) > 0, M(Choi L4) > 0, M(Bell) = 0; random depth-3 forward states whose
     three Paulis have rank-3 ancilla X-parts have mana exactly 3 M(m) (= E0 (x) m^3 up to Clifford),
     random rank-3 backward states from E0 R have mana M(Choi R) + 3 M(m).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "4")
import numpy as np, itertools
rng = np.random.default_rng(1)
z = np.exp(2j*np.pi/9); w = z**3
X = np.roll(np.eye(3), 1, axis=0); Z = np.diag([1, w, w*w]); I3 = np.eye(3)
H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(3)
mp = np.linalg.matrix_power
N = 4; DIM = 81
def kron(*ops):
    out = np.ones((1, 1))
    for o in ops: out = np.kron(out, o)
    return out
def pauli(a, b): return kron(*[mp(X, a[i]) @ mp(Z, b[i]) for i in range(len(a))])
def f_of(P):
    """f(P) for a (phased) Pauli with P^3 = I: eigenvalue w^k -> (1, z, z^8)[k]."""
    d = P.shape[0]; fv = [1, z, z**8]; Pk = [np.eye(d), P, P @ P]; out = np.zeros((d, d), complex)
    for m in range(3): out += fv[m]*sum(w**(-m*k)*Pk[k] for k in range(3))/3
    return out
ket0 = np.array([1, 0, 0], complex)

# ---------------- (1) magic states ----------------
def wigner_mana(psi, n):
    """mana log sum|W| of pure state psi on n qutrits (odd-d Gross Wigner function)."""
    A0 = np.zeros((3, 3)); A0[[0, 2, 1], [0, 1, 2]] = 1          # parity |-x><x|
    A = np.array([mp(X, q) @ mp(Z, p) @ A0 @ np.conj(mp(X, q) @ mp(Z, p)).T for q in range(3) for p in range(3)])
    R = np.outer(psi, np.conj(psi))
    Rt = R.reshape([3]*n + [3]*n)
    for k in range(n):
        # Rt axes: (u_0..u_{k-1}, i_k..i_{n-1}, j_k..j_{n-1}) ; contract i_k, j_k with A[u, j, i] (Tr A rho = sum A_ji rho_ij)
        nu = k; ni = n - k
        Rt = np.tensordot(Rt, A, axes=([nu, nu + ni], [2, 1]))   # removes i_k, j_k, appends u_k at end
        # move u_k to position k
        Rt = np.moveaxis(Rt, -1, k)
    W = Rt.real.reshape(-1)/3**n
    assert abs(W.sum() - 1) < 1e-9
    return np.log(np.abs(W).sum())

def is_nonstab_1q(m):
    return max(abs(np.vdot(m, mp(X, a) @ mp(Z, b) @ m)) for a in range(3) for b in range(3) if (a, b) != (0, 0))

mags = {}
for c in range(3):
    F = f_of(w**c*X)
    mags[f"m_{c}"] = F @ ket0; mags[f"mbar_{c}"] = np.conj(F).T @ ket0
print("(1) single-qutrit magic states:")
Mm = None
for k, m in mags.items():
    mm = wigner_mana(m, 1); mx = is_nonstab_1q(m)
    print(f"    {k}: mana {mm:.12f}   max_(P!=I)|<P>| = {mx:.6f}")
    assert mx < 1 - 1e-6
    Mm = mm if Mm is None else Mm; assert abs(mm - Mm) < 1e-12

# ---------------- (2) fresh-step lemma ----------------
def M2(q): # |x> -> |2x> on qutrit q
    U = np.zeros((3, 3)); U[[0, 2, 1], [0, 1, 2]] = 1
    return kron(*[U if i == q else I3 for i in range(N)])
def S(q): return kron(*[np.diag([1, 1, w]) if i == q else I3 for i in range(N)])
def SUM(c, t):
    U = np.zeros((DIM, DIM))
    for x in itertools.product(range(3), repeat=N):
        y = list(x); y[t] = (y[t] + y[c]) % 3
        U[int(np.dot(y, 3**np.arange(N)[::-1])), int(np.dot(x, 3**np.arange(N)[::-1]))] = 1
    return U
def CZ(c, t):
    d = np.array([w**(x[c]*x[t]) for x in itertools.product(range(3), repeat=N)]); return np.diag(d)
LABELS = list(itertools.product(range(3), repeat=2*N))
PAULI = {lab: pauli(lab[:N], lab[N:]) for lab in LABELS}
def decompose(P):
    """P = w^c X^a Z^b  ->  (a, b, c), read off columns (X^a Z^b |x> = w^{b.x} |x+a>), then checked."""
    PW = 3**np.arange(N)[::-1]
    col0 = P[:, 0]; i0 = int(np.argmax(abs(col0)))
    a = np.array([(i0 // PW[k]) % 3 for k in range(N)])
    c = int(np.rint(np.angle(col0[i0])/(2*np.pi/3))) % 3
    b = np.zeros(N, int)
    for k in range(N):
        col = P[:, PW[k]]; ii = int(np.argmax(abs(col)))
        b[k] = (int(np.rint(np.angle(col[ii])/(2*np.pi/3))) - c) % 3
    assert np.allclose(P, w**c*pauli(a, b)); return a, b, c
GATES = {}
for j in range(3):
    for k in (1, 2):
        GATES[('SUM', j, k)] = mp(SUM(3, j), k); GATES[('CZ', j, k)] = mp(CZ(3, j), k)
for k in (1, 2): GATES[('S', k)] = mp(S(3), k)
GATES[('M2',)] = M2(3)
def canonicalise(P):
    E = np.eye(DIM, dtype=complex)
    def app(G):
        nonlocal P, E
        P = G @ P @ np.conj(G).T; E = G @ E
    a, b, c = decompose(P)
    if a[3] == 2: app(GATES[('M2',)])
    for j in range(3):
        for kind, idx in (('SUM', 0), ('CZ', 1)):
            a, b, c = decompose(P)
            if (a if idx == 0 else b)[j] != 0:
                for k in (1, 2):
                    G = GATES[(kind, j, k)]; a2, b2, _ = decompose(G @ P @ np.conj(G).T)
                    if (a2 if idx == 0 else b2)[j] == 0 and a2[3] == 1: app(G); break
                else: raise RuntimeError
    a, b, c = decompose(P)
    if b[3] != 0:
        for k in (1, 2):
            G = GATES[('S', k)]; a2, b2, _ = decompose(G @ P @ np.conj(G).T)
            if b2[3] == 0: app(G); break
    a, b, c = decompose(P)
    assert list(a) == [0, 0, 0, 1] and list(b) == [0, 0, 0, 0], (a, b)
    return E, c

def rand_parent(kind):
    """27x3 parent isometries on q0..q2."""
    E00 = np.zeros((27, 3), complex)
    for x in range(3): E00[9*x, x] = 1
    J = E00 @ (np.diag([1, 1, -1]) if kind == 'R' else np.eye(3))
    for _ in range({'E0': 0, 'R': 2, 'd1': 1, 'd3': 3}[kind]):
        a = rng.integers(0, 3, 3); b = rng.integers(0, 3, 3)
        if not (a.any() or b.any()): a[0] = 1
        J = f_of(pauli(a, b)) @ J
    return J
parents = [rand_parent(k) for k in ('E0', 'd1', 'd3', 'R', 'd3')]
nf = nn = 0
for lab in LABELS:
    a = np.array(lab[:N]); b = np.array(lab[N:])
    if not (a.any() or b.any()): continue
    P = PAULI[lab]; FP = f_of(P)
    if a[3] != 0:
        E, c = canonicalise(P.copy())
        mc = f_of(w**c*X) @ ket0
        for Jt in parents:
            Jp = np.kron(Jt, ket0[:, None])
            assert np.allclose(E @ Jp, Jp)                                     # E fixes the parent
            assert np.allclose(FP @ Jp, np.conj(E).T @ np.kron(Jt, mc[:, None]))  # child ~ Jt (x) m_c
        nf += 1
    else:
        Pt = pauli(a[:3], b[:3])
        for Jt in parents:
            Jp = np.kron(Jt, ket0[:, None])
            assert np.allclose(FP @ Jp, np.kron(f_of(Pt) @ Jt, ket0[:, None]))
        nn += 1
print(f"(2) fresh-step lemma verified for {nf} fresh and {nn} non-fresh Paulis x {len(parents)} parents")

# ---------------- (3) mana of Choi states ----------------
def choi(J):     # J: (3^n) x 3 isometry -> normalised Choi vector on ref (x) out
    return (J.T.reshape(-1))/np.sqrt(3)   # ordering ref first
R = np.diag([1, 1, -1]); L4 = np.diag([1, 1, z]); T1 = np.diag([1, z, z**8])
for name, G in (("I", np.eye(3)), ("T", T1), ("R", R), ("L4", L4)):
    print(f"(3) mana(Choi {name}) = {wigner_mana(choi(G), 2):.12f}")
MR = wigner_mana(choi(R), 2); ML4 = wigner_mana(choi(L4), 2)
assert MR > 1e-3 and ML4 > 1e-3
E0 = np.zeros((DIM, 3), complex)
for x in range(3): E0[27*x, x] = 1
def rand_rank3_state(start, inverse=False):
    while True:
        Ps = [(rng.integers(0, 3, N), rng.integers(0, 3, N)) for _ in range(3)]
        Ax = np.array([a[1:] for a, b in Ps])
        if round(abs(np.linalg.det(Ax))) % 3 != 0: break        # det != 0 mod 3  <=> rank 3 over F3
    J = start.copy()
    for a, b in Ps:
        F = f_of(pauli(a, b)); J = (np.conj(F).T if inverse else F) @ J
    return J
for trial in range(4):
    mf = wigner_mana(choi(rand_rank3_state(E0)), 5)
    mb = wigner_mana(choi(rand_rank3_state(E0 @ R, inverse=True)), 5)
    mb4 = wigner_mana(choi(rand_rank3_state(E0 @ L4, inverse=True)), 5)
    print(f"    rank-3 fwd: {mf:.10f} (3M(m)={3*Mm:.10f});  bwd R: {mb:.10f} (MR+3M(m)={MR+3*Mm:.10f});"
          f"  bwd L4: {mb4:.10f} (ML4+3M(m)={ML4+3*Mm:.10f})")
    assert abs(mf - 3*Mm) < 1e-9 and abs(mb - MR - 3*Mm) < 1e-9 and abs(mb4 - ML4 - 3*Mm) < 1e-9
print("ALL CHECKS PASSED")
