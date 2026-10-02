"""tmerge.py — exact T-rotation merging on the emitted (system + 1 shared ancilla) circuit.

Writes U = K · R_m ··· R_1 with R_k = B_k^dag T^{±1}_{q_k} B_k (9x9, exact up to float),
then scans each new rotation backwards past commuting rotations (symplectic test on axes);
if it meets a rotation with a dependent axis (P or P^2) the two are multiplied in place.
A merged element f(P) is classified by its eigenphases on the P-eigenspaces:
Clifford (0 T) or T-type (1 T).  The rotation product is re-checked against U, so the
reduced T-count is that of an explicit, exactly equal 2-qutrit Clifford+T circuit
(each surviving T-type f(P) = C T C^dag with C a Clifford sending Z_q -> P).
Also returns the rot-depth of the merged rotation list."""
import numpy as np
from tdepth import Gate, parse, local_symplectic_of_dagger, symp, _dependent, ONE

W = np.exp(2j * np.pi / 3)

def _embed2(g):
    if len(g.qubits) == 1:
        return np.kron(g.U, np.eye(3)) if g.qubits[0] == 0 else np.kron(np.eye(3), g.U)
    if g.qubits == (0, 1):
        return g.U
    S = np.zeros((9, 9))
    for x in range(3):
        for y in range(3):
            S[3 * y + x, 3 * x + y] = 1
    return S @ g.U @ S

Z1 = np.diag([1, W, W * W])
ZQ = {0: np.kron(Z1, np.eye(3)), 1: np.kron(np.eye(3), Z1)}

def _cost(M, P):
    ph = []
    for k in range(3):
        Pi = sum(W ** (-k * s) * np.linalg.matrix_power(P, s) for s in range(3)) / 3
        ph.append(np.trace(M @ Pi) / np.trace(Pi))
    e = [np.angle(p / ph[0]) * 9 / (2 * np.pi) for p in ph]
    ei = [int(round(x)) % 9 for x in e]
    assert all(abs(x - round(x)) < 1e-6 for x in e), e
    if all(x % 3 == 0 for x in ei):
        return 0
    assert sum(ei) % 3 == 0, ei          # T-type
    return 1

def merge(gates):
    gs = [g for g in gates if g.kind in ("C", "T")]
    assert all(q in (0, 1) for g in gs for q in g.qubits)
    n = 2
    N = np.eye(4, dtype=np.int64)
    B = np.eye(9, dtype=complex)
    U = np.eye(9, dtype=complex)
    els = []                       # [M, P, axis, cost]
    for g in gs:
        Gm = _embed2(g)
        U = Gm @ U
        if g.kind == "C":
            loc = list(g.qubits)
            S = local_symplectic_of_dagger(g.U)
            cols = loc + [n + i for i in loc]
            N[:, cols] = (N[:, cols] @ S) % 3
            B = Gm @ B
            continue
        q = g.qubits[0]
        R = B.conj().T @ _embed2(g) @ B
        P = B.conj().T @ ZQ[q] @ B
        ax = N[:, n + q].copy()
        k = len(els) - 1
        placed = False
        while k >= 0:
            if _dependent(els[k][2], ax):
                els[k][0] = R @ els[k][0]
                els[k][3] = _cost(els[k][0], els[k][1])
                placed = True
                break
            if symp(els[k][2], ax, n):
                break
            k -= 1
        if not placed:
            els.append([R, P, ax, 1])
    # verify  U == B · prod(els)
    Prod = np.eye(9, dtype=complex)
    for M, *_ in els:
        Prod = M @ Prod
    err = np.abs(B @ Prod - U).max()
    tc = sum(e[3] for e in els)
    # depth of merged list (Clifford elements contribute 0 and do not block)
    live = [e for e in els if e[3]]
    depth = []
    for j, e in enumerate(live):
        d = max([depth[i] for i in range(j) if symp(live[i][2], e[2], n)], default=0)
        depth.append(d + 1)
    return dict(T_merged=tc, rot_depth_merged=max(depth, default=0), merge_err=float(err))


def merge_symbolic(gates):
    """Same merge rule on the symplectic picture only (any number of qutrits, measurement aware).
    Element exponent e in Z_3: T(P)^e (e=0 -> Clifford).  T(P^2) = T(P)^dag; phases in P only add
    Cliffords.  Measurements / feed-forward block a backward scan exactly as in tdepth.rot_depth."""
    from tdepth import rotations
    nodes, n, idx = rotations(gates)
    els = []        # [kind, axis, exponent] ; kind T or M or F
    for kind, ax, sign, extra in nodes:
        if kind == "T":
            k = len(els) - 1
            placed = False
            while k >= 0:
                ek, eax, ee = els[k]
                if ek == "F":
                    sysq = [idx[q] for q in ee]
                    if any(ax[q] % 3 for q in sysq):
                        break
                elif ek == "T" and _dependent(eax, ax):
                    s = sign if np.array_equal(eax % 3, ax % 3) else -sign
                    els[k][2] = (ee + s) % 3
                    placed = True
                    break
                elif symp(eax, ax, n):
                    break
                k -= 1
            if not placed:
                els.append(["T", ax, sign % 3])
        elif kind == "M":
            els.append(["M", ax, None])
        else:
            els.append(["F", None, extra])
    return sum(1 for e in els if e[0] == "T" and e[2] % 3)
