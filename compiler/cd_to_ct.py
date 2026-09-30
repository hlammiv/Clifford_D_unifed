"""cd_to_ct.py — emit an explicit qutrit Clifford+T circuit (one clean ancilla)
from a Clifford+D exact decomposition, and verify it.

Input: an exact unitary V over Z[zeta_9, 1/3] and its canonical_reducer
decomposition (syllables P_i = H·Diag(ζ^a0, ζ^a1, ζ^a2)·R^eps·X^delta and the
trailing monomial C, with V = P_1† ··· P_n† · C).

Output: a time-ordered gate list on qutrit 0 (system) and qutrit 1 (clean
ancilla, |0> in and out), over the gate set
    Clifford:  single-qutrit Clifford matrices ("C0", named or explicit), SUM01, SUM10
    non-Clifford: T0/T1, Tdg0/Tdg1 with T = diag(1, ζ, ζ^8)
Non-Clifford syllable pieces are expanded as
    T-type diagonal (level 3)    -> T or T† on the system            (1 T)
    level-4 diagonal             -> C2X ; T/T†(ancilla) ; C2X†        (7 T; unified/level4/l4_7T_verify.py)
    R = diag(1,1,-1)             -> 63-gate 7-T word                  (7 T; unified/r_from_d/R_7T_compact.json)
Each gadget acts on computational position j of the system via X^(2-j) conjugation.

verify_circuit() checks, numerically, that U (psi ⊗ |0>) = e^{i phi} (V psi) ⊗ |0>
for random psi, that every non-T gate is Clifford, and returns the T-count.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent
sys.path[:0] = [str(_UNIFIED), str(_UNIFIED / "hrsa")]

import canonical_reducer as cr  # noqa: E402

Z9 = np.exp(2j * np.pi / 9)
W = Z9 ** 3
I3 = np.eye(3, dtype=complex)
H = np.array([[W ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(-3 + 0j)
X = np.roll(np.eye(3), 1, axis=0).astype(complex)
T = np.diag([1, Z9, Z9 ** 8])
dg = lambda A: A.conj().T  # noqa: E731
ONE = {"H": H, "Hdg": dg(H), "X": X, "Xdg": dg(X), "Z": np.diag([1, W, W * W]),
       "S": np.diag([1, 1, W]), "Sdg": dg(np.diag([1, 1, W])), "T": T, "Tdg": dg(T)}
SUM01 = np.zeros((9, 9))
SUM10 = np.zeros((9, 9))
for _x in range(3):
    for _y in range(3):
        SUM01[3 * _x + (_y + _x) % 3, 3 * _x + _y] = 1   # |x,y> -> |x, x+y>
        SUM10[3 * ((_x + _y) % 3) + _y, 3 * _x + _y] = 1  # |x,y> -> |x+y, y>

# ---------------------------------------------------------------- gadgets
C2X = ["H1", "X0", "X1"] + ["Xdg1", "T1", "X1", "SUM01"] * 3 + ["Xdg0", "Xdg1", "Hdg1"]


def _inverse(word):
    inv = {"H": "Hdg", "Hdg": "H", "X": "Xdg", "Xdg": "X", "T": "Tdg", "Tdg": "T",
           "S": "Sdg", "Sdg": "S"}
    out = []
    for g in reversed(word):
        if g in ("SUM01", "SUMdg01"):
            out.append("SUMdg01" if g == "SUM01" else "SUM01")
        elif g in ("SUM10", "SUMdg10"):
            out.append("SUMdg10" if g == "SUM10" else "SUM10")
        elif g[:-1] == "Z":
            out += [g, g]
        else:
            out.append(inv[g[:-1]] + g[-1])
    return out


def _l4_word(j: int, sign: int):
    """diag with ζ^{sign} at system position j (sign = ±1), 7 T, clean ancilla."""
    sh = (2 - j) % 3
    pre = ["X0"] * sh
    post = ["Xdg0"] * sh
    return pre + C2X + (["T1"] if sign == 1 else ["Tdg1"]) + _inverse(C2X) + post


def _load_R_word():
    d = json.load(open(_UNIFIED / "r_from_d" / "R_7T_compact.json"))
    return [g if q is None else f"{g}{q}" for g, q in d["gates"]]


_R_WORD = _load_R_word()


def _r_word(j: int):
    """diag with -1 at system position j, 7 T, clean ancilla (up to global phase)."""
    sh = (2 - j) % 3
    return ["X0"] * sh + _R_WORD + ["Xdg0"] * sh


# ---------------------------------------------------------------- ring -> complex
_BASIS = np.array([Z9 ** k for k in range(6)])


def _c(zf) -> complex:
    return complex(np.dot([int(c) for c in zf.num.coefs], _BASIS) / 3 ** zf.denom_pow3)


def ring_to_complex(M):
    return np.array([[_c(M[i][j]) for j in range(3)] for i in range(3)])


# ---------------------------------------------------------------- diagonal expansion
def _expand_diag(exps):
    """Gate list for diag(ζ^e0, ζ^e1, ζ^e2) up to global phase.
    Returns (gates, n_t3, n_l4)."""
    e = [x % 9 for x in exps]
    tvec = {0: (0, 0, 0), 1: (0, 1, 8), 2: (0, 8, 1)}
    best = None
    for g, m, sgn, j in itertools.product(range(9), range(3), (0, 1, -1), range(3)):
        if sgn == 0 and j:
            continue
        rem = [(e[i] - g - tvec[m][i] - (sgn if i == j else 0)) % 9 for i in range(3)]
        if all(r % 3 == 0 for r in rem):
            cost = (1 if m else 0) + (7 if sgn else 0)
            if best is None or cost < best[0]:
                best = (cost, m, sgn, j, rem)
    assert best is not None
    _, m, sgn, j, rem = best
    gates = []
    if m:
        gates.append("T0" if m == 1 else "Tdg0")
    if sgn:
        gates += _l4_word(j, sgn)
    cl = np.diag([Z9 ** r for r in rem])          # ω-diagonal: Clifford
    if not np.allclose(cl, I3):
        gates.append(("C0", cl))
    return gates, (1 if m else 0), (1 if sgn else 0)


def _unit_parts(u: complex):
    for k in range(9):
        if abs(u - Z9 ** k) < 1e-9:
            return 1, k
        if abs(u + Z9 ** k) < 1e-9:
            return -1, k
    raise ValueError(f"not a unit ±ζ^k: {u}")


def emit_circuit(syllables, trailing):
    """Time-ordered gate list for V = P_1† ··· P_n† · C.
    Returns (gates, counts) with counts = dict(n_T3, n_L4, n_R)."""
    counts = dict(n_T3=0, n_L4=0, n_R=0)
    gates = []
    # --- trailing monomial C = diag(d) · Perm  (Perm applied first)
    Cc = ring_to_complex(trailing)
    Pm = (np.abs(Cc) > 0.5).astype(complex)
    d = np.array([Cc[i][np.argmax(np.abs(Cc[i]))] for i in range(3)])
    gates.append(("C0", Pm))
    parts = [_unit_parts(u) for u in d]
    signs = [s for s, _ in parts]
    ks = [k for _, k in parts]
    if len(set(signs)) > 1:                          # mixed signs -> one R
        odd = [i for i in range(3) if signs.count(signs[i]) == 1][0]
        gates += _r_word(odd)
        counts["n_R"] += 1
    g, t3, l4 = _expand_diag(ks)
    gates += g
    counts["n_T3"] += t3
    counts["n_L4"] += l4
    # --- syllables, last one applied first:  P† = X^-δ · R^ε · D(-a) · H†
    for s in reversed(syllables):
        gates.append("Hdg0")
        g, t3, l4 = _expand_diag([-s["a0"], -s["a1"], -s["a2"]])
        gates += g
        counts["n_T3"] += t3
        counts["n_L4"] += l4
        if s["eps"]:
            gates += _r_word(2)
            counts["n_R"] += 1
        gates += ["Xdg0"] * (s["delta"] % 3)
    return gates, counts


# ---------------------------------------------------------------- verification
def _gate_matrix(gt):
    if isinstance(gt, tuple):
        name, A = gt
        return np.kron(A, I3) if name == "C0" else np.kron(I3, A)
    if gt == "SUM01":
        return SUM01
    if gt == "SUMdg01":
        return dg(SUM01)
    if gt == "SUM10":
        return SUM10
    if gt == "SUMdg10":
        return dg(SUM10)
    base, q = gt[:-1], int(gt[-1])
    return np.kron(ONE[base], I3) if q == 0 else np.kron(I3, ONE[base])


_PAULIS = [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(ONE["Z"], b)
           for a in range(3) for b in range(3)]


def _is_clifford_1q(A) -> bool:
    for P in (X, ONE["Z"]):
        Q = A @ P @ dg(A)
        if not any(abs(abs(np.vdot(R, Q)) / 3 - 1) < 1e-9 for R in _PAULIS):
            return False
    return True


def tcount(gates) -> int:
    return sum(1 for g in gates if isinstance(g, str) and g[:-1] in ("T", "Tdg"))


def verify_circuit(gates, V, n_states: int = 50, seed: int = 0):
    """Return (max_err, T-count).  Raises if a non-T gate is not Clifford."""
    for g in gates:
        if isinstance(g, tuple):
            assert _is_clifford_1q(g[1]), "explicit single-qutrit gate is not Clifford"
    U = np.eye(9, dtype=complex)
    for g in gates:
        U = _gate_matrix(g) @ U
    Vc = V if isinstance(V, np.ndarray) else ring_to_complex(V)
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n_states):
        psi = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi /= np.linalg.norm(psi)
        out = U @ np.kron(psi, [1, 0, 0])
        tgt = np.kron(Vc @ psi, [1, 0, 0])
        ph = np.vdot(tgt, out)
        err = max(err, float(np.linalg.norm(out - ph * tgt)), abs(abs(ph) - 1))
    return err, tcount(gates)


def compile_matrix(V, select: str = "tcost"):
    """Decompose V with canonical_reducer (prefix selection `select`) and emit
    the verified Clifford+T circuit.  Returns dict with gates and stats."""
    cr.set_selection_cost(select)
    r = cr.decompose_canonical(V)
    if not r["success"]:
        raise RuntimeError(r.get("error"))
    gates, counts = emit_circuit(r["syllables"], r["trailing_clifford"])
    err, tc = verify_circuit(gates, V)
    return dict(gates=gates, err=err, tcount=tc, n_gates=len(gates),
                formula_tcost=counts["n_T3"] + 7 * counts["n_L4"] + 7 * counts["n_R"],
                N_D=r["D_count"], **counts)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(_UNIFIED / "nick_test"))
    from ingest_decompose import parse_fits_file, build_ring
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--select", choices=["d", "tcost"], default="tcost")
    a = ap.parse_args()
    f, rows = parse_fits_file(_UNIFIED / "nick_test" / f"fits_f={a.f}.txt")
    for _, th, g in rows[: a.n]:
        res = compile_matrix(build_ring(g, f), a.select)
        tgt = np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2), 1])
        eps = np.linalg.norm(ring_to_complex(build_ring(g, f)) - tgt)
        print(f"f={f} θ={th:.4f} ε={eps:.2e}  gates={res['n_gates']:5d}  T-count={res['tcount']:4d} "
              f"(formula {res['formula_tcost']:4d}; T3 {res['n_T3']} L4 {res['n_L4']} R {res['n_R']})  "
              f"N_D={res['N_D']}  verify err={res['err']:.1e}")
