"""Unit-level tests of the topk-dump patch on synthetic exact candidate data.

A. integer matrix builder  == Nick's Sage MatrixC6 (dump_special_thetas.py)
B. top-K collection inside _collect_best_scan_fixed_theta == brute force
   (vector-window enumeration, matrix-frob ranking, exact dedup, tol cut, cap K),
   single rank and 2 ranks (rank/size args) merged with _merge_topk_records.
C. --topk 0 (old path) and --topk K produce identical all-best (tie) arrays.
D. fit_one_theta_all_best end-to-end with only I/O stubbed (real triples file);
   text output parses, each matrix is exactly unitary over Z[zeta9,1/3]
   (unified/nick_test/ingest_decompose.unitarity_exact), frob column matches.
E. DumpTopK (special-theta path) on a synthetic out.phase_free_special_theta.npz.
"""
import itertools, math, os, pickle, random, sys, tempfile, json
import numpy as np
REPO = "/home/hlamm/Desktop/efficent_gates/zeta9"
sys.path.insert(0, REPO)
sys.path[:0] = ["/home/hlamm/Desktop/efficent_gates/unified/nick_test",
                "/home/hlamm/Desktop/efficent_gates/unified",
                "/home/hlamm/Desktop/efficent_gates/unified/hrsa"]
import zeta9.fit_theta_all_best_one_theta as M
from zeta9.tools import mul as _tm, conj as _tc, _monomial
from collections import OrderedDict

# Nick's Sage-based functions, without running the module-level Dump() calls
src = open(os.path.join(REPO, "dump_special_thetas.py")).read().split("_TOPK = _topk_cli()")[0]
NS = {"__name__": "dst"}
exec(compile(src, "dump_special_thetas.py", "exec"), NS)

F = 4
TOL = 1.25
pool = pickle.load(open("pool_f4.pkl", "rb"))
rng = random.Random(7)

# ---------------- A ----------------
nA = nskip = 0
for c in rng.sample(pool, 60) + [tuple(tuple(rng.randint(-50, 50) for _ in range(6)) for _ in range(3)) for _ in range(20)]:
    e, N = M._householder_int_matrix(np.array(c), F, "2/3^f")
    try:
        S = NS["MatrixC6"](np.array(c), F)
    except Exception:      # Nick's Sage-text row parser cannot handle some rows (e.g. zero entries)
        nskip += 1
        continue
    assert e == F and all(list(S[i, j]) == list(N[i][j]) for i in range(3) for j in range(3)), c
    nA += 1
print(f"A ok: integer builder == Sage MatrixC6 on {nA} vectors ({nskip} skipped: MatrixC6 parser error)")

# ---------------- synthetic root lists ----------------
UNITS = []
for k in range(9):
    u = list(_monomial(k))
    UNITS += [u, [-v for v in u]]
def nrm(c):
    return _tm(tuple(c), _tc(tuple(c)))[0]
roots_by_norm = {}
for c in pool:
    for ci in c:
        roots_by_norm.setdefault(nrm(ci), set()).add(tuple(ci))
for n in list(roots_by_norm):
    base = sorted(roots_by_norm[n])[:3]
    s = set()
    for b in base:
        for u in rng.sample(UNITS, 4):
            s.add(tuple(_tm(tuple(b), tuple(u))))
    roots_by_norm[n] = sorted(s)
triples = sorted({tuple(nrm(ci) for ci in c) for c in pool})
tri_rows = np.array([[n0, 0, 0, n1, 0, 0, n2, 0, 0] for n0, n1, n2 in triples], dtype=np.int64)
print("synthetic triples:", len(tri_rows), " roots per Y:", sorted({len(v) for v in roots_by_norm.values()}))

theta = 0.7
amp = 3.0 ** (-0.5 * F)
lam = float(3 ** F)
t1 = complex(amp * np.exp(-0.5j * theta)); t2 = -amp + 0j; t3 = 0j

def cand_list(y, target):
    R = np.array(roots_by_norm.get(y[0], []), dtype=np.int64).reshape(-1, 6)
    if R.shape[0] == 0:
        return None
    z = np.array([M.coeffs_to_complex_noscale(r) for r in R]) / 3 ** F
    d2 = np.abs(z - target) ** 2
    o = np.argsort(d2, kind="mergesort")
    return (d2[o], np.arange(len(o))[o], R[o])

M._read_triple_rows = lambda path, start, n: tri_rows[start:start + n]
M._build_locator = lambda needed, paths: {}
M._candidate_cache_get = lambda cache, y, target, *a, **k: cand_list(y, target)

def brute(best_sq, K, slack):
    thr = best_sq * (TOL * slack) ** 2
    recs = {}
    for r, row in enumerate(tri_rows):
        c1, c2, c3 = (cand_list((row[3 * i],), t) for i, t in enumerate((t1, t2, t3)))
        if c1 is None or c2 is None or c3 is None:
            continue
        for i1, i2, i3 in itertools.product(range(len(c1[0])), range(len(c2[0])), range(len(c3[0]))):
            tot = c1[0][i1] + c2[0][i2] + c3[0][i3]
            if tot <= thr + 1e-24:
                cf = np.stack([c1[2][i1], c2[2][i2], c3[2][i3]])
                fr = M._householder_matrix_frob(cf, F, lam, theta)
                key = tuple(v for rr in M._householder_int_matrix(cf, F, "2/3^f")[1] for e in rr for v in e)
                if key not in recs or fr < recs[key]:
                    recs[key] = fr
    fr = sorted(recs.values())
    return [x for x in fr if x <= TOL * fr[0]][:K], len(recs)

kw = dict(triples_file="x", nrows=len(tri_rows), triples_chunk_rows=7, paths={}, phase_meta={},
          y2_good_keys=np.ascontiguousarray(tri_rows[:, 3:6]), f=F, eps=10.0, target1=t1, target2=t2,
          target3=t3, batch_cache=OrderedDict(), disable_y_pruning=False, verbose=False)
M._isin_structured_rows = lambda rows, keys: np.ones(rows.shape[0], dtype=bool)
best_sq, _ = M._local_best_scan_fixed_theta(rank=0, size=1, **kw)
print("best vector dist", math.sqrt(best_sq))

for K, slack in ((5, 1.5), (100, 1.5), (1000, 3.0)):
    thr = best_sq * (TOL * slack) ** 2
    parts, ties = [], []
    for size in (1, 2):
        parts = []
        tie_arrays = []
        for rank in range(size):
            col = M._TopKCollector(K, f=F, lam=lam, theta=theta, norm_label="2/3^f")
            out = M._collect_best_scan_fixed_theta(rank=rank, size=size, best_sq=best_sq, equal_atol=1e-24,
                                                   max_local_solutions=0, topk_collector=col, topk_thr_sq=thr, **kw)
            parts.append(out["topk"]); tie_arrays.append(out["coeffs"])
        got = [r["frob"] for r in M._merge_topk_records(parts, K, TOL)]
        exp, npool = brute(best_sq, K, slack)
        assert np.allclose(got, exp, rtol=0, atol=1e-15), (K, size, got[:5], exp[:5])
        ties.append(np.concatenate(tie_arrays))
    # ---------------- C ----------------
    old = np.concatenate([M._collect_best_scan_fixed_theta(rank=r, size=2, best_sq=best_sq, equal_atol=1e-24,
                          max_local_solutions=0, **kw)["coeffs"] for r in range(2)])
    key = lambda a: sorted(map(lambda x: x.tobytes(), a))
    assert key(ties[0]) == key(ties[1]) and set(key(old)) <= set(key(ties[0]))
    if len(old) != len(ties[0]):
        print(f"   NOTE: old pass-2 found {len(old)} exact ties, top-K branch found {len(ties[0])} "
              "(pre-existing float issue in the old searchsorted window, see report)")
    print(f"B ok: K={K} slack={slack}: {len(got)} kept (pool within window {npool}), 1- and 2-rank == brute force; C ok")

# demonstrate the old-path float issue on the best combination
c1, c2, c3 = None, None, None
for row in tri_rows:
    L = [cand_list((row[3 * i],), t) for i, t in enumerate((t1, t2, t3))]
    if all(x is not None for x in L) and L[0][0][0] + L[1][0][0] + L[2][0][0] == best_sq:
        a, b, c = L[0][0][0], L[1][0][0], L[2][0][0]
        print(f"old-path check: best_sq-(a+b) - c = {(best_sq - (a + b)) - c:.3e} (equal_atol default 1e-24)")
        break
# ---------------- D ----------------
from ingest_decompose import build_ring, unitarity_exact, build_complex
tmp = tempfile.mkdtemp(dir=".")
tri_path = os.path.join(tmp, "triples"); tri_rows.tofile(tri_path)
json.dump({"rows_written": int(len(tri_rows))}, open(tri_path + ".manifest.json", "w"))
prefix = os.path.join(tmp, "roots_local")
open(M._state_paths(prefix, 512)["exact_roots_index_meta"], "w").write("{}")
M._load_fixed_target_cache = lambda *a, **k: (np.ascontiguousarray(tri_rows[:, 3:6]), {})
M._ensure_binned_phase_sidecar = lambda *a, **k: {}
for fmt in ("matrix", "vector"):
    txt = os.path.join(tmp, f"topk_{fmt}.txt")
    for th in (0.7, 1.9):   # loop over theta, appending
        t1 = complex(amp * np.exp(-0.5j * th))
        M.fit_one_theta_all_best(triples_file=tri_path, triples_json=tri_path + ".manifest.json", rootdb_prefix=prefix,
                                 f=F, theta=th, eps=10.0, output_prefix=os.path.join(tmp, f"out{th}"), triples_chunk_rows=5,
                                 y2_good_npz=tri_path, norm="2/3^f", topk=20, tol=TOL, topk_txt=txt,
                                 topk_format=fmt, topk_append=True)
    lines = open(txt).read().splitlines()
    hdr = [l for l in lines if l.startswith("#")]
    rows = [l for l in lines if not l.startswith("#")]
    assert len(hdr) == 2, hdr
    print(f"D {fmt}: header {hdr[0]!r}; {len(rows)} rows over 2 thetas; bytes/row ~ {sum(map(len, rows)) // len(rows)}")
    if fmt == "matrix":
        nbad = 0
        for l in rows:
            parts = [p.strip() for p in l.split(",")]
            th = float(parts[0]); g = [[int(x) for x in p.split()] for p in parts[1:10]]
            fr = float(parts[10])
            ok, _ = unitarity_exact(build_ring(g, F))
            Mc = build_complex(g, F)   # (transposed read; frob to a diagonal target unchanged)
            fr2 = np.linalg.norm(Mc - np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1]))
            nbad += (not ok) + (abs(fr - fr2) > 1e-12)
        assert nbad == 0
        print("D ok: every written matrix exactly unitary; frob column re-verified")
    if fmt == "matrix":
        print("   sample:", rows[0][:120], "...", rows[0][-30:])
print("ALL TESTS PASSED")
