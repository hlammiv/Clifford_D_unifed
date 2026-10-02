"""tdepth.py — T-depth calculator for qutrit Clifford+T gate lists.

Gate-list formats accepted (all time-ordered):
  * cd_to_ct.py:   "H0","Tdg1","SUM01","SUMdg01","SUM10", ("C0", 3x3 Clifford matrix), ("C1", ...)
  * R_7T_compact.json: [name, q] pairs, q=None for SUM01/SUM10
  * measurement JSONs: "SUM01dg", "Sdg1", ...
Internally a gate is Gate(kind, qubits, U) with kind in {"T","C","M"}; T gates carry
sign (+1 for T, -1 for Tdg).  Qutrit indices can be remapped (fresh ancillas).

Three T-depth measures:
  wire_depth      : as written; gates block each other iff they share a qutrit
                    (Cliffords cost 0 depth but synchronise their qutrits).
  rot_depth       : Pauli-rotation DAG.  Every T is pushed to a Clifford-conjugated
                    rotation T(P) (P = B^dag Z_q B, B = Clifford prefix).  Edge i->j iff
                    the axes do not commute (symplectic product != 0).  Longest chain.
                    This is the minimum T-depth over all re-orderings that keep the
                    multiset of rotations (any commuting layer can be executed at depth 1
                    given enough ancillas, Tpar-style).  Measurements (meas. model) are
                    nodes of weight 0: they depend on non-commuting earlier rotations
                    and every later rotation whose axis is touched by the feed-forward
                    correction (or does not commute with the measured Pauli) depends on them.
  mergeable       : number of rotation pairs with dependent axes (P vs P or P^2) that can
                    be brought adjacent (no non-commuting rotation between) -> T-count
                    could drop (T(P)T(P)=C*T(P)^dag, T(P)T(P)^dag = I).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

Z9 = np.exp(2j * np.pi / 9)
W = Z9 ** 3
I3 = np.eye(3, dtype=complex)
H = np.array([[W ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(-3 + 0j)
Xm = np.roll(np.eye(3), 1, axis=0).astype(complex)
Zm = np.diag([1, W, W * W])
T = np.diag([1, Z9, Z9 ** 8])
dg = lambda A: A.conj().T  # noqa: E731
ONE = {"H": H, "Hdg": dg(H), "X": Xm, "Xdg": dg(Xm), "Z": Zm, "Zdg": dg(Zm),
       "S": np.diag([1, 1, W]), "Sdg": dg(np.diag([1, 1, W])), "T": T, "Tdg": dg(T)}
SUM = np.zeros((9, 9), dtype=complex)          # |x,y> -> |x, x+y>  (first = control)
for _x, _y in itertools.product(range(3), repeat=2):
    SUM[3 * _x + (_x + _y) % 3, 3 * _x + _y] = 1


@dataclass
class Gate:
    kind: str            # "T", "C", "M" (measurement of qubits[0]; basis "Z" or "F")
    qubits: tuple
    U: np.ndarray | None = None
    sign: int = 0
    basis: str = "Z"


def parse(g, qmap=None) -> Gate:
    """Normalise one gate of any supported format.  qmap remaps local qutrit labels."""
    qm = (lambda q: qmap.get(q, q)) if qmap else (lambda q: q)
    if isinstance(g, Gate):
        return Gate(g.kind, tuple(qm(q) for q in g.qubits), g.U, g.sign, g.basis)
    if isinstance(g, (list, tuple)) and len(g) == 2 and isinstance(g[0], str) and not isinstance(g[1], np.ndarray):
        name, q = g                                   # R_7T_compact.json pair
        g = name if q is None else f"{name}{q}"
    if isinstance(g, tuple):                          # ("C0", A)
        name, A = g
        return Gate("C", (qm(int(name[1])),), np.asarray(A, dtype=complex))
    s = g.replace("SUMdg", "SUM").replace("dg", "dg")
    if g.startswith("SUM"):
        inv = "dg" in g
        digits = "".join(ch for ch in g if ch.isdigit())
        c, t = int(digits[0]), int(digits[1])         # SUMct: control c, target t
        U = dg(SUM) if inv else SUM
        return Gate("C", (qm(c), qm(t)), U)
    base, q = s[:-1], qm(int(s[-1]))
    if base in ("T", "Tdg"):
        return Gate("T", (q,), ONE[base], +1 if base == "T" else -1)
    return Gate("C", (q,), ONE[base])


# ------------------------------------------------------------------ Pauli algebra
_P1 = {(a, b): np.linalg.matrix_power(Xm, a) @ np.linalg.matrix_power(Zm, b)
       for a in range(3) for b in range(3)}


def _local_pauli(xs, zs):
    M = np.array([[1]], dtype=complex)
    for a, b in zip(xs, zs):
        M = np.kron(M, _P1[(a, b)])
    return M


def _decompose(Q, k):
    d = 3 ** k
    for xs in itertools.product(range(3), repeat=k):
        for zs in itertools.product(range(3), repeat=k):
            if abs(abs(np.vdot(_local_pauli(xs, zs), Q)) - d) < 1e-7:
                return np.array(list(xs) + list(zs))
    raise ValueError("not a Pauli: gate is not Clifford")


_SCACHE: dict = {}


def local_symplectic_of_dagger(U):
    """S with columns = Pauli vectors of U^dag G U for generators G = X_i.., Z_i.. ."""
    key = (U.shape[0], np.round(U, 8).tobytes())
    if key in _SCACHE:
        return _SCACHE[key]
    k = 1 if U.shape[0] == 3 else 2
    cols = []
    for which in ("x", "z"):
        for i in range(k):
            xs = [0] * k
            zs = [0] * k
            (xs if which == "x" else zs)[i] = 1
            G = _local_pauli(xs, zs)
            cols.append(_decompose(dg(U) @ G @ U, k))
    S = np.array(cols).T % 3
    _SCACHE[key] = S
    return S


def symp(u, v, n):
    return int((u[:n] @ v[n:] - u[n:] @ v[:n]) % 3)


def _dependent(u, v):
    return np.array_equal(u % 3, v % 3) or np.array_equal(u % 3, (2 * v) % 3)


# ------------------------------------------------------------------ depth measures
def normalise(gates, qmap=None):
    return [parse(g, qmap) for g in gates]


def wire_depth(gates):
    gs = [g if isinstance(g, Gate) else parse(g) for g in gates]
    t = {}
    for g in gs:
        cur = max(t.get(q, 0) for q in g.qubits)
        if g.kind == "T":
            cur += 1
        for q in g.qubits:
            t[q] = cur
    return max(t.values()) if t else 0


def rotations(gates):
    """Return list of nodes (kind, axis vector, sign, extra) in time order on n qutrits."""
    gs = [g if isinstance(g, Gate) else parse(g) for g in gates]
    qs = sorted({q for g in gs for q in g.qubits})
    idx = {q: i for i, q in enumerate(qs)}
    n = len(qs)
    N = np.eye(2 * n, dtype=np.int64)
    nodes = []
    for g in gs:
        loc = [idx[q] for q in g.qubits]
        if g.kind == "C":
            S = local_symplectic_of_dagger(g.U)
            cols = loc + [n + i for i in loc]
            N[:, cols] = (N[:, cols] @ S) % 3
        elif g.kind == "T":
            nodes.append(("T", N[:, n + loc[0]].copy(), g.sign, None))
        elif g.kind == "M":
            i = loc[0]
            ax = N[:, n + i] if g.basis == "Z" else N[:, i]   # Fourier basis measures X
            nodes.append(("M", ax.copy(), 0, None))
        elif g.kind == "F":                               # feed-forward correction marker
            nodes.append(("F", None, 0, g.qubits))
    return nodes, n, idx


def rot_depth(gates, return_details=False):
    nodes, n, idx = rotations(gates)
    depth = []
    pending_meas = []        # indices of measurement nodes whose correction is pending/applied
    corr_dep = []            # (meas node, system index) -> later rotations with X on that qutrit depend
    best = 0
    for j, (kind, ax, sign, extra) in enumerate(nodes):
        if kind == "F":
            depth.append(depth[-1] if depth else 0)
            corr_dep.append((pending_meas.pop(), [idx[q] for q in extra]))
            continue
        d = 0
        for i in range(j):
            ki, ai, _, _ = nodes[i]
            if ki == "F":
                continue
            if symp(ai, ax, n):
                d = max(d, depth[i])
        for (mi, sysq) in corr_dep:
            if any(ax[q] % 3 for q in sysq) or kind == "M":
                d = max(d, depth[mi])
        val = d + (1 if kind == "T" else 0)
        depth.append(val)
        if kind == "M":
            pending_meas.append(j)
        best = max(best, val)
    if not return_details:
        return best
    return best, nodes, depth


def mergeable_pairs(gates):
    """Count T-rotation pairs i<j with dependent axes and no non-commuting rotation between."""
    nodes, n, _ = rotations(gates)
    tn = [(k, a, s) for k, a, s, _ in nodes if k == "T"]
    cnt = 0
    used = set()
    for i in range(len(tn)):
        if i in used:
            continue
        for j in range(i + 1, len(tn)):
            if j in used:
                continue
            if _dependent(tn[i][1], tn[j][1]):
                if all(symp(tn[k][1], tn[i][1], n) == 0 for k in range(i + 1, j)):
                    cnt += 1
                    used |= {i, j}
                break
            if symp(tn[j][1], tn[i][1], n):
                break
    return cnt


def tcount(gates):
    return sum(1 for g in gates if (g if isinstance(g, Gate) else parse(g)).kind == "T")
