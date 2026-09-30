"""analyze_candidates.py — R-count study over many exact approximants per theta.

Input: HRSA_tester logs (HRSA_bestD with --max-solns N) containing CANDDUMP lines:
  CANDDUMP s D x1re x1im x2re x2im x3re x3im f  n1[6] n2[6] n3[6]
where u_i = n_i / 3^f (power basis 1..zeta^5), V = P01 (I - u u^dag)  (decompose_impl.h buildUnitary).

For each distinct candidate V: decompose with canonical_reducer (strict; greedy fallback noted),
record frob, N_D, #R syllables, level-3/level-4 diagonal counts, trailing-monomial sign pattern,
a det-sign parity invariant, and chi-adic first-column residue features.

Repo code is NOT modified.
Usage: python3 analyze_candidates.py raw/hrsa_*.log --out candidates.csv [--jobs N]
"""
from __future__ import annotations
import argparse, csv, re, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "hrsa"), str(_HERE.parent / "nick_test")]
import canonical_reducer as cr  # noqa: E402

Z = np.exp(2j * np.pi / 9)
BASIS = np.array([Z ** k for k in range(6)])


def target(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2), 1.0])


# ---------------- exact helpers ----------------
def fz(coefs, f):
    return cr._FastZ9Frac(cr._FastZ9(tuple(cr._INT(int(c)) for c in coefs)), f)


def build_V(n1, n2, n3, f):
    u = [fz(n1, f), fz(n2, f), fz(n3, f)]
    one = cr._FastZ9Frac.one(); zero = cr._FastZ9Frac.zero()
    Hh = [[(one if i == j else zero) - u[i] * u[j].conjugate() for j in range(3)] for i in range(3)]
    return [Hh[1][:], Hh[0][:], Hh[2][:]]


def to_complex(zf):
    return complex(np.dot([int(c) for c in zf.num.coefs], BASIS) / 3 ** zf.denom_pow3)


def Vc(V):
    return np.array([[to_complex(V[i][j]) for j in range(3)] for i in range(3)])


def scale(V, zf):
    return [[zf * V[i][j] for j in range(3)] for i in range(3)]


def unit_sign(x: complex):
    """x = s * zeta^k (s=+-1) -> s."""
    for k in range(9):
        if abs(x - Z ** k) < 1e-6:
            return 1
        if abs(x + Z ** k) < 1e-6:
            return -1
    return 0


# chi-adic leading digit: c = prod_{k in 2,4,5,7,8}(1-zeta^k) = 3/chi exactly
_C = None


def _cz():
    global _C
    if _C is None:
        one = cr._zeta9_power(0).num
        c = one
        for k in (2, 4, 5, 7, 8):
            c = c * (one - cr._zeta9_power(k).num)
        _C = c
    return _C


def chi_digit(coefs):
    """Return (v, d): v = chi-adic valuation of N in Z[zeta], d = (N/chi^v) mod chi in {1,2}."""
    N = cr._FastZ9(tuple(cr._INT(int(c)) for c in coefs))
    if N.is_zero():
        return 999, 0
    v = 0
    while int(sum(int(c) for c in N.coefs)) % 3 == 0:
        M = N * _cz()
        assert all(int(c) % 3 == 0 for c in M.coefs)
        N = cr._FastZ9(tuple(cr._INT(int(c) // 3) for c in M.coefs))
        v += 1
    return v, int(sum(int(c) for c in N.coefs)) % 3


def col_signature(V, col=0):
    """chi-adic signature of column `col`: relative valuations and leading digits of the
    min-valuation entries, normalized up to global sign (digit of first min entry := 1)."""
    ent = []
    for i in range(3):
        e = V[i][col]
        v, d = chi_digit(e.num.coefs)
        vv = 999 if v == 999 else v - 6 * e.denom_pow3  # chi-adic valuation of full element
        ent.append((vv, d))
    vmin = min(x[0] for x in ent)
    rel = tuple((x[0] - vmin) if x[0] < 900 else 9 for x in ent)
    digs = [x[1] if x[0] == vmin else 0 for x in ent]
    ref = next(d for d in digs if d)
    digs = tuple((d * (1 if ref == 1 else 2)) % 3 for d in digs)
    return rel, digs


def det_sign(V):
    return unit_sign(np.linalg.det(Vc(V)))


# ---------------- decomposition ----------------
def analyze_syllables(res):
    syl = res["syllables"]
    nR = sum(s["eps"] for s in syl)
    lvl3 = lvl4 = cliff = 0
    for s in syl:
        a = (s["a0"], s["a1"], s["a2"])
        if all(x % 3 == 0 for x in a):
            cliff += 1
        elif sum(a) % 3 == 0:
            lvl3 += 1
        else:
            lvl4 += 1
    return nR, lvl3, lvl4, cliff


def trailing_info(T):
    Tc = Vc(T)
    signs = []
    for i in range(3):
        j = int(np.argmax(abs(Tc[i])))
        signs.append(unit_sign(Tc[i, j]))
    resid_R = 0 if len(set(signs)) == 1 else 1
    return signs, resid_R, unit_sign(np.linalg.det(Tc))


_DETH = None
BACKEND = "py"
_PROC = None
DECOMP_TOOL = str(_HERE.parent / "hrsa" / "decompose_tool")


def V_to_json(V):
    import json
    f = max(V[i][j].denom_pow3 for i in range(3) for j in range(3))
    M = [[[int(c) * 3 ** (f - V[i][j].denom_pow3) for c in V[i][j].num.coefs]
          for j in range(3)] for i in range(3)]
    return json.dumps({"f": f, "V": M})


def cpp_decompose(V):
    """C++ decompose_tool (same sde-peeling algorithm; validated vs canonical_reducer on a subset)."""
    import json, subprocess
    global _PROC
    if _PROC is None:
        _PROC = subprocess.Popen([DECOMP_TOOL, "--persistent"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    _PROC.stdin.write(V_to_json(V) + "\n"); _PROC.stdin.flush()
    line = _PROC.stdout.readline()
    while line.strip() and not line.lstrip().startswith("{"):
        line = _PROC.stdout.readline()
    r = json.loads(line)
    if r.get("success"):
        tc = r["trailing_clifford"]
        r["trailing_clifford"] = [[fz(tc["V"][i][j], tc["f"]) for j in range(3)] for i in range(3)]
        r["sde_chi_initial"] = r.get("sde_chi_initial", -1)
    return r


def work(item):
    global _DETH
    key, theta, eps, f, n1, n2, n3, hrsa_D = item
    V = build_V(n1, n2, n3, f)
    M = Vc(V)
    frob = float(np.linalg.norm(M - target(theta)))
    unit_err = float(np.linalg.norm(M @ M.conj().T - np.eye(3)))
    t0 = time.time()
    greedy = False
    if BACKEND == "cpp":
        res = cpp_decompose(V)
    else:
        res = cr.decompose_canonical(V, greedy_single=False)
        if not res["success"]:
            res = cr.decompose_canonical(V, greedy_single=True); greedy = True
    row = dict(theta=theta, eps=eps, f=f, key=key, frob=frob, unit_err=unit_err,
               hrsa_D=hrsa_D, success=res["success"], greedy=greedy,
               dec_s=round(time.time() - t0, 3))
    if res["success"]:
        nR, l3, l4, cl = analyze_syllables(res)
        _, _, mr = cr.classify_monomial_and_d_cost(res["trailing_clifford"])
        signs, rR, tdet = trailing_info(res["trailing_clifford"])
        syl = res["syllables"]
        row.update(N_D=res["D_count"], n_syl=len(syl), R_syl=nR, R_resid_classify=mr,
                   R_resid_sign=rR, R_total=nR + rR, lvl3=l3, lvl4=l4, cliff_diag=cl,
                   first_eps=syl[0]["eps"] if syl else -1,
                   R_positions=" ".join(str(i) for i, s in enumerate(syl) if s["eps"]),
                   trail_signs="".join("+" if s > 0 else "-" for s in signs), trail_det=tdet,
                   sde0=res["sde_chi_initial"])
    rel, digs = col_signature(V, 0)
    row.update(col0_rel="".join(map(str, rel)), col0_dig="".join(map(str, digs)),
               det_sign=det_sign(V))
    rel1, digs1 = col_signature([list(r) for r in zip(*V)], 0)  # row 0 signature
    row.update(row0_rel="".join(map(str, rel1)), row0_dig="".join(map(str, digs1)))
    return row


def parse_log(path):
    m = re.search(r"hrsa_t([0-9.]+)_e([0-9.e-]+)\.log", path.name)
    theta, eps = float(m.group(1)), float(m.group(2))
    out, seen = [], set()
    for line in open(path):
        if not line.startswith("CANDDUMP"):
            continue
        p = line.split()
        s, D, f = int(p[1]), int(p[2]), int(p[9])
        ints = [int(x) for x in p[10:28]]
        n1, n2, n3 = ints[0:6], ints[6:12], ints[12:18]
        key = (tuple(n1), tuple(n2), tuple(n3), f)
        if key in seen:
            continue
        seen.add(key)
        out.append((f"{path.stem}#{s}", theta, eps, f, n1, n2, n3, D))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=str(_HERE / "candidates.csv"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--backend", choices=["py", "cpp"], default="py")
    ap.add_argument("--limit", type=int, default=0, help="random sample (seed 12345) of this many candidates per log (0=all)")
    a = ap.parse_args()
    global BACKEND
    BACKEND = a.backend
    items = []
    for lg in a.logs:
        it = parse_log(Path(lg))
        if a.limit and len(it) > a.limit:
            import random
            it = random.Random(12345).sample(it, a.limit)
        items += it
    print(f"{len(items)} distinct candidates", flush=True)
    cr.get_prefix_table()
    with Pool(a.jobs) as pool:
        rows = pool.map(work, items, chunksize=4)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
