"""mana_bounds.py -- magic-monotone lower bounds valid in the MEASUREMENT + feed-forward and CATALYST models.
Any Clifford+T circuit with t T-gates (adaptive, with measurements, Clifford feed-forward, any number of
stabilizer ancillas, and a returned catalyst c) is a stabilizer protocol consuming t copies of |T> = T|+>
(T is level 3: injected with Clifford corrections).  Mana M (log of Wigner negativity) is additive and
non-increasing under stabilizer protocols, so applying the gadget to half of a maximally entangled state
(stabilizer, M=0):   M(c) + t M(T|+>) >= M(c) + M(Choi(G))   =>   t >= M(Choi(G)) / M(T|+>)   (catalyst cancels).
Same with the best stabilizer input |s>:   t >= M(G|s>)/M(T|+>).  For RUS the bound holds for the
expectation (mana is monotone on average? NO -- only for trace-preserving maps; we state it for
deterministic gadgets, with or without catalyst)."""
import numpy as np, itertools
z = np.exp(2j*np.pi/9); w = z**3
X = np.roll(np.eye(3), 1, axis=0); Zm = np.diag([1, w, w*w]); mp = np.linalg.matrix_power
def D(a1, a2): return w**(2*a1*a2) * mp(Zm, a1) @ mp(X, a2)
P0 = np.zeros((3, 3))
for j in range(3): P0[(-j) % 3, j] = 1
A = {(a1, a2): D(a1, a2) @ P0 @ D(a1, a2).conj().T for a1 in range(3) for a2 in range(3)}
def wig(rho, n):
    out = []
    for pts in itertools.product(list(A), repeat=n):
        Op = np.array([[1]])
        for p in pts: Op = np.kron(Op, A[p])
        out.append(np.trace(Op @ rho).real / 3**n)
    return np.array(out)
def mana(psi, n):
    psi = np.asarray(psi, complex); psi = psi/np.linalg.norm(psi); W = wig(np.outer(psi, psi.conj()), n)
    assert abs(W.sum()-1) < 1e-9
    return np.log(np.abs(W).sum())
MT = mana([1, z, z**8], 1)
print(f"M(T|+>) = {MT:.6f}   (sanity: stabilizer |0> has M = {mana([1,0,0],1):.1e})")
H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(3)
S = np.diag([1, 1, w])
# the 12 single-qutrit stabilizer states
stabs = []
for U in [np.eye(3), H, H @ S, H @ S @ S]:
    for k in range(3): stabs.append(U @ np.eye(3)[:, k])
for name, G in [('R = diag(1,1,-1)', np.diag([1, 1, -1])), ('L = diag(1,1,zeta)', np.diag([1, 1, z])),
                ('T (sanity, must give 1)', np.diag([1, z, z**8]))]:
    best1 = max(mana(G @ s, 1) for s in stabs)
    choi = np.zeros(9, complex)
    for x in range(3): choi[3*x+x] = G[x, x]
    mc = mana(choi, 2)
    print(f"{name:26s} max_s M(G|s>) = {best1:.6f} -> t >= {best1/MT:.4f};  M(Choi) = {mc:.6f} -> t >= {mc/MT:.4f}")
