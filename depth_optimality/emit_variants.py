"""emit_variants.py — re-emit cd_to_ct circuits in three variants and measure T-depth.
  shared : exactly cd_to_ct.emit_circuit (one ancilla reused)            [checked identical]
  fresh  : same gates, each 7-T gadget on its own fresh clean ancilla     (relabeling only)
  meas   : measurement model, L4 = R = 4 T (l_meas_4T / r_meas_4T), fresh ancilla per gadget,
           random outcomes + Clifford feed-forward; one branch per run is simulated exactly
           (system + 1 re-prepared ancilla) to check correctness.
Usage: python3 emit_variants.py --f 4 --n 8 [--seed 0]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; U_ = HERE.parent
sys.path[:0] = [str(U_ / "compiler"), str(U_ / "nick_test"), str(HERE)]
import cd_to_ct as cc
from ingest_decompose import parse_fits_file, build_ring
from tdepth import Gate, parse, normalise, wire_depth, rot_depth, mergeable_pairs, tcount

LM = json.load(open(U_ / "measurement_tricks/l_meas_4T.json"))
RM = json.load(open(U_ / "measurement_tricks/r_meas_4T.json"))

class Emitter:
    def __init__(self, mode, rng):
        self.mode, self.rng, self.next_anc = mode, rng, 1
    def anc(self):
        if self.mode == "shared":
            return 1
        self.next_anc += 1
        return self.next_anc - 1
    def _corr(self, exps):
        return Gate("C", (0,), np.diag([cc.W ** e for e in exps]).astype(complex))
    def l4(self, j, sign):
        if self.mode != "meas":
            return normalise(cc._l4_word(j, sign), {1: self.anc()})
        a = self.anc(); sh = (2 - j) % 3
        gl = list(LM["gates"]); 
        if sign == -1:
            gl[-1] = "Tdg1"
        m = int(self.rng.integers(3))
        c = LM["correction_exponents"][str(m)]
        if sign == -1:
            c = list(c)   # same correction for Tdg (phase left is w^{-m b} either way)
        M = Gate("M", (a,), basis="F"); M.m = m
        return (normalise(["X0"] * sh + gl, {1: a}) + [M, self._corr(c),
                Gate("F", (0,))] + normalise(["Xdg0"] * sh))
    def r(self, j):
        if self.mode != "meas":
            return normalise(cc._r_word(j), {1: self.anc()})
        a = self.anc(); sh = (2 - j) % 3
        m = int(self.rng.integers(3))
        M = Gate("M", (a,), basis="Z"); M.m = m
        return (normalise(["X0"] * sh + RM["gates"], {1: a}) + [M,
                self._corr(RM["correction_exponents"][str(m)]), Gate("F", (0,))] + normalise(["Xdg0"] * sh))
    def diag(self, exps):
        g, t3, l4 = cc._expand_diag(exps)          # reuse cd_to_ct's choice, then swap the gadget
        out = []
        i = 0
        while i < len(g):
            x = g[i]
            if isinstance(x, str) and x[-1] == "1" or x == "SUM01" or (isinstance(x, str) and x == "X0" and l4):
                break
            out.append(parse(x)); i += 1
        if l4:
            # recover (j, sign) from cd_to_ct's word: leading X0 count and middle T1/Tdg1
            sh = 0
            while g[i + sh] == "X0":
                sh += 1
            j = (2 - sh) % 3
            mid = g[i + sh + len(cc.C2X)]
            sign = 1 if mid == "T1" else -1
            word_len = 2 * sh + 2 * len(cc.C2X) + 1
            out += self.l4(j, sign)
            out += normalise(g[i + word_len:])
        return out, t3, l4

def emit(syllables, trailing, mode, rng):
    E = Emitter(mode, rng)
    counts = dict(n_T3=0, n_L4=0, n_R=0)
    gates = []
    Cc = cc.ring_to_complex(trailing)
    Pm = (np.abs(Cc) > 0.5).astype(complex)
    d = np.array([Cc[i][np.argmax(np.abs(Cc[i]))] for i in range(3)])
    gates.append(parse(("C0", Pm)))
    parts = [cc._unit_parts(u) for u in d]
    signs = [s for s, _ in parts]; ks = [k for _, k in parts]
    if len(set(signs)) > 1:
        odd = [i for i in range(3) if signs.count(signs[i]) == 1][0]
        gates += E.r(odd); counts["n_R"] += 1
    g, t3, l4 = E.diag(ks); gates += g; counts["n_T3"] += t3; counts["n_L4"] += l4
    for s in reversed(syllables):
        gates.append(parse("Hdg0"))
        g, t3, l4 = E.diag([-s["a0"], -s["a1"], -s["a2"]])
        gates += g; counts["n_T3"] += t3; counts["n_L4"] += l4
        if s["eps"]:
            gates += E.r(2); counts["n_R"] += 1
        gates += normalise(["Xdg0"] * (s["delta"] % 3))
    return gates, counts

def simulate_branch(gates, Vc, rng, n_states=10):
    """Exact branch simulation: system + ONE ancilla, re-prepared |0> after each measurement
    (every gadget ancilla starts clean, so relabeling to one physical ancilla is valid).
    Projects on the outcome the emitter drew; the correction is already in the list."""
    F = np.array([[cc.W ** (j * mm) for mm in range(3)] for j in range(3)]) / np.sqrt(3)
    err = 0.0
    for _ in range(n_states):
        psi = rng.normal(size=3) + 1j * rng.normal(size=3); psi /= np.linalg.norm(psi)
        st = np.kron(psi, [1, 0, 0]).astype(complex)
        for g in gates:
            if g.kind == "M":
                A = st.reshape(3, 3)
                if g.basis == "F":
                    A = A @ F.conj()
                v = A[:, g.m]
                p = np.vdot(v, v).real
                assert abs(p - 1 / 3) < 1e-6, p
                st = np.kron(v / np.sqrt(p), [1, 0, 0]).astype(complex)
                continue
            if g.kind == "F":
                continue
            qs = tuple(0 if q == 0 else 1 for q in g.qubits)
            if len(qs) == 1:
                M = np.kron(g.U, cc.I3) if qs[0] == 0 else np.kron(cc.I3, g.U)
            else:
                M = g.U if qs == (0, 1) else SWAP @ g.U @ SWAP
            st = M @ st
        tgt = np.kron(Vc @ psi, [1, 0, 0])
        ph = np.vdot(tgt, st)
        err = max(err, float(np.linalg.norm(st - ph * tgt)))
    return err

SWAP = np.zeros((9, 9))
for x in range(3):
    for y in range(3):
        SWAP[3 * y + x, 3 * x + y] = 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--f", type=int, default=4); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    f, rows = parse_fits_file(U_ / "nick_test" / f"fits_f={a.f}.txt")
    pick = np.linspace(0, len(rows) - 1, a.n).astype(int)
    out = []
    for i in pick:
        _, th, g = rows[i]
        V = build_ring(g, f)
        cc.cr.set_selection_cost("tcost")
        r = cc.cr.decompose_canonical(V)
        ref, cnt = cc.emit_circuit(r["syllables"], r["trailing_clifford"])
        Vc = cc.ring_to_complex(V)
        row = dict(f=f, theta=th, n_T3=cnt["n_T3"], n_L4=cnt["n_L4"], n_R=cnt["n_R"],
                   n_syl=len(r["syllables"]))
        for mode in ("shared", "fresh", "meas"):
            gl, c2 = emit(r["syllables"], r["trailing_clifford"], mode, rng)
            if mode == "shared":
                assert len(gl) == len(ref) and all(
                    (parse(x).kind == y.kind and parse(x).qubits == y.qubits and np.allclose(parse(x).U, y.U))
                    for x, y in zip(ref, gl)), "shared re-emit differs from cd_to_ct"
            row[f"T_{mode}"] = tcount(gl)
            row[f"wire_{mode}"] = wire_depth([x for x in gl if x.kind != "F"]) if mode != "meas" else wire_depth([x for x in gl if x.kind not in "F"])
            row[f"rot_{mode}"] = rot_depth(gl)
            if mode == "shared":
                row["merge_shared"] = mergeable_pairs(gl)
            if mode == "meas":
                row["sim_err_meas"] = simulate_branch(gl, Vc, rng)
                row["n_anc_meas"] = len({q for x in gl for q in x.qubits}) - 1
        out.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
