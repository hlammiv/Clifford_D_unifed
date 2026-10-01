"""E. DumpTopK (special-theta path, dump_special_thetas.py --topk) on a synthetic
out.phase_free_special_theta.npz built from exact vectors; compare with Dump()."""
import os, sys, pickle, subprocess, tempfile
import numpy as np
REPO = "/home/hlamm/Desktop/efficent_gates/zeta9"
sys.path.insert(0, REPO)
sys.path[:0] = ["/home/hlamm/Desktop/efficent_gates/unified/nick_test", "/home/hlamm/Desktop/efficent_gates/unified",
                "/home/hlamm/Desktop/efficent_gates/unified/hrsa"]
src = open(os.path.join(REPO, "dump_special_thetas.py")).read().split("_TOPK = _topk_cli()")[0]
NS = {"__name__": "dst"}; exec(compile(src, "dst", "exec"), NS)
from zeta9.tools import embed
from ingest_decompose import build_ring, unitarity_exact
F = 4
pool = pickle.load(open("pool_f4.pkl", "rb"))
good = pool[:150]
th, c1, c2, c3, gf = [], [], [], [], []
for c in good:
    z1, z2 = embed(c[0], 1), embed(c[1], 1)
    if abs(z1) == 0 or abs(z2) == 0:
        continue
    d1 = -np.conj(z2 / abs(z2)) * (z1 / abs(z1))
    th.append(float(np.mod(-2 * np.angle(d1), 4 * np.pi))); c1.append(c[0]); c2.append(c[1]); c3.append(c[2]); gf.append(0.0)
n = len(th)
work = tempfile.mkdtemp(dir=".")
d = os.path.join(work, "D/V_f=4/eps=1e-2"); os.makedirs(d)
np.savez(os.path.join(d, "out.phase_free_special_theta.npz"), theta=np.array(th), theta_off=np.arange(n + 1),
         group_frob=np.array(gf), group_coeff3=np.array(c3, dtype=np.int64),
         coeff1=np.array(c1, dtype=np.int64), coeff2=np.array(c2, dtype=np.int64))
env = dict(os.environ, PYTHONPATH=REPO)
r = subprocess.run([sys.executable, os.path.join(REPO, "dump_special_thetas.py"), "--topk", "10", "--tol", "1.25", "--fs", "4"],
                   cwd=work, env=env, capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-300:])
from zeta9.tools import householder_int_matrix
old = {}   # theta -> the matrix that generated it (what Dump() writes; Dump's Sage parser chokes on these sparse synthetic rows)
for t, a1, a2, a3 in zip(th, c1, c2, c3):
    N = householder_int_matrix(np.array([a1, a2, a3]), F)[1]
    old.setdefault(t, [" ".join(str(v) for v in N[i][j]) for j in range(3) for i in range(3)])
rows = [l for l in open(os.path.join(work, "fits_topk_f=4.txt")) if not l.startswith("#")]
nbad = 0; ncontains = 0; nshould = 0; byth = {}
for l in rows:
    p = [x.strip() for x in l.split(",")]
    t = float(p[0]); g = [[int(x) for x in q.split()] for q in p[1:10]]
    nbad += not unitarity_exact(build_ring(g, F))[0]
    byth.setdefault(t, []).append((float(p[10]), int(p[11]), p[1:10]))
for t, L in byth.items():
    fr = [x[0] for x in L]
    assert fr == sorted(fr) and fr[-1] <= 1.25 * fr[0] + 1e-15 and [x[1] for x in L] == list(range(len(L)))
    present = (t in old) and any(x[2] == old[t] for x in L)
    ncontains += present
    if t in old:
        g = [[int(v) for v in q.split()] for q in old[t]]
        from ingest_decompose import build_complex
        gfr = np.linalg.norm(build_complex(g, F) - np.diag([np.exp(-0.5j*t), np.exp(0.5j*t), 1]))
        should = gfr <= 1.25 * fr[0] and sum(x[0] < gfr - 1e-15 for x in L) < 10
        assert present == should or (not present and len(L) == 10), (t, gfr, fr)
        nshould += should
print(f"E: {len(rows)} rows over {len(byth)} thetas (Dump wrote {len(old)}); non-unitary={nbad}; "
      f"sorted/tol/rank ok; generating matrix present for {ncontains}/{len(byth)} thetas = exactly those where it lies within tol*best ({nshould})")
assert nbad == 0
