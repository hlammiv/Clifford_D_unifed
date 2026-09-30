"""Exact density-matrix simulation of the 2-to-1 stabilizer reductions used to turn
qutrit strange states |S> into R-states (Prakash 2003.02717 App. A; Anwar et al.
1202.2326 Sec. IV), plus a check of the QRM_3(2) 8->1 T-state map
(Campbell-Anwar-Browne 1205.3104).

All operations used here are Pauli/Clifford measurements + postselection + Clifford
decoding: NO non-Clifford gate is consumed by the conversions.
"""
import itertools
import numpy as np

w = np.exp(2j * np.pi / 3)
X = np.roll(np.eye(3), 1, axis=0)          # X|j> = |j+1>
Z = np.diag([1, w, w**2])
I3 = np.eye(3)
k = np.kron


def proj_from_stab(g):
    """Projector onto +1 eigenspace of an order-3 operator g."""
    return (np.eye(g.shape[0]) + g + g @ g) / 3


def ket(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


S = ket([0, 1, -1])            # strange state (H eigenstate, eigenvalue i)
Nst = ket([0, 1, 1])           # Norell state


def depol(psi, delta):
    return (1 - delta) * np.outer(psi, psi.conj()) + delta * I3 / 3


# ---- stage 1: |S>|S> -> |N>, stabilizer Z1 Z2, logical Zbar = Z2, Xbar = X1^2 X2
P1 = proj_from_stab(k(Z, Z))
# logical basis |jbar> = |-j, j>
V1 = np.zeros((3, 9), dtype=complex)
for j in range(3):
    V1[j, ((-j) % 3) * 3 + j] = 1.0      # decode |-j,j> -> |j>


# ---- stage 2: |N>|N> -> R-state, stabilizer omega X1 X2, Xbar = X2, Zbar = Z1^2 Z2
P2 = proj_from_stab(w * k(X, X))
Zb = k(Z @ Z, Z)
Xb = k(I3, X)
ev, evec = np.linalg.eigh(P2)
code = evec[:, ev > 0.5]                  # 9x3 basis of code space
# |0bar>: Zbar eigenvalue 1 inside code space
zc = code.conj().T @ Zb @ code
e2, u2 = np.linalg.eig(zc)
i0 = np.argmin(np.abs(e2 - 1))
b0 = code @ u2[:, i0]
b0 /= np.linalg.norm(b0)
basis2 = [b0, Xb @ b0, Xb @ Xb @ b0]
V2 = np.array([b.conj() for b in basis2])  # 3x9 decoding isometry


def reduce2(rho1, rho2, P, V):
    r = P @ k(rho1, rho2) @ P
    p = np.real(np.trace(r))
    out = V @ r @ V.conj().T
    return out / p, p


def R_state_from_S(delta):
    """Return (rho_R, p1, p2, ideal_R) starting from two stages of depolarised |S>."""
    rs = depol(S, delta)
    rN, p1 = reduce2(rs, rs, P1, V1)
    rR, p2 = reduce2(rN, rN, P2, V2)
    return rN, rR, p1, p2


rN0, rR0, p10, p20 = R_state_from_S(0.0)
N_ideal = np.linalg.eigh(rN0)[1][:, -1]
R_ideal = np.linalg.eigh(rR0)[1][:, -1]


def infid(rho, psi):
    return 1 - np.real(psi.conj() @ rho @ psi)


def R_state_infidelity(delta):
    rN, rR, p1, p2 = R_state_from_S(delta)
    return infid(rN, N_ideal), infid(rR, R_ideal), p1, p2


# ---- equatorialisation of Anwar et al.: |Psi+>|Psi+> -> (|0>+|1>-|2>)/sqrt3
def equatorialisation_prob():
    psi = ket([1, 1, 0])
    L = [ket(np.kron(np.eye(3)[0], np.eye(3)[0]) + w * np.kron(np.eye(3)[1], np.eye(3)[2])
             + w**2 * np.kron(np.eye(3)[2], np.eye(3)[1]))]
    L.append(k(X, X) @ L[0])
    L.append(k(X, X) @ L[1])
    two = np.kron(psi, psi)
    amps = np.array([l.conj() @ two for l in L])
    return np.sum(np.abs(amps) ** 2), amps / np.linalg.norm(amps)


# ---- R-state is Clifford-equivalent to (1,1,-1)/sqrt3 ? check via |amplitudes| and
# eigen-structure of the injected diagonal gate (phases are what matter).
def describe_R(psi):
    ph = psi / psi[np.argmax(np.abs(psi))]
    return np.abs(psi) ** 2, np.angle(ph) / np.pi


# ---- QRM_3(2) exact depolarising map via weight enumerators (CAB Eq. preMacWill)
u = np.array([[1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 1, 2, 2, 2]])
vZ = np.array([[1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1, 2, 0, 2, 1, 0],
               [1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]])


def span_weights(G):
    ws = []
    for c in itertools.product(range(3), repeat=G.shape[0]):
        v = (np.array(c) @ G) % 3
        ws.append(int(np.count_nonzero(v)))
    return np.bincount(ws, minlength=9)


def dual_weights(G):
    ws = []
    for v in itertools.product(range(3), repeat=G.shape[1]):
        if np.all((G @ np.array(v)) % 3 == 0):
            ws.append(int(np.count_nonzero(v)))
    return np.bincount(ws, minlength=9)


A_LZ = span_weights(vZ)
A_LXp = dual_weights(u)


def qrm_map(eps):
    """Exact QRM_3(2) 8->1 under depolarising (twirled) noise: returns (eps_out, P_succ)."""
    f0 = 1 - eps
    fj = eps / 2
    num = sum(A_LZ[w_] * f0 ** (8 - w_) * fj ** w_ for w_ in range(9))
    den = sum(A_LXp[w_] * f0 ** (8 - w_) * fj ** w_ for w_ in range(9))
    return 1 - num / den, den


if __name__ == "__main__":
    print("QRM weight enumerators  L_Z:", A_LZ, " L_X^perp:", A_LXp)
    lo, hi = 0.1, 0.3
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if qrm_map(mid)[0] < mid else (lo, mid)
    print(f"QRM depolarising threshold = {lo:.6f}  (CAB: 0.211001)")
    for e in [1e-2, 1e-3]:
        eo, P = qrm_map(e)
        print(f"  eps={e:g}: eps_out={eo:.3e}  eps_out/eps^2={eo/e**2:.3f}  P={P:.4f}")
    print("pure-state conversion probs: S S->N p=%.4f, N N->R p=%.4f" % (p10, p20))
    print("ideal R-state |amp|^2, phases/pi:", describe_R(R_ideal))
    pe, amps = equatorialisation_prob()
    print("Anwar equatorialisation success prob = %.4f, output amps" % pe, np.round(amps, 4))
    for d in [1e-3, 1e-4, 1e-6, 1e-8]:
        eN, eR, p1, p2 = R_state_infidelity(d)
        eS = 2 * d / 3
        print(f"delta_S={d:g} (infid {eS:.2e}): eps_N={eN:.3e} ({eN/eS:.3f} eps_S), "
              f"eps_R={eR:.3e} ({eR/eS:.3f} eps_S), p1={p1:.4f}, p2={p2:.4f}")
