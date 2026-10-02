"""Kernel-level check: fast new_s vector == original per-prefix sde for every prefix."""
import sys, random
from pathlib import Path
U = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parent), str(U / "nick_test"), str(U), str(U / "hrsa")]
import numpy as np
import canonical_reducer as cr
import fast_decompose as fd
from ingest_decompose import parse_fits_file, build_ring
tbl = cr.get_prefix_table()
assert [ (e.a1,e.a2,e.a3,e.eps,e.delta,e.d) for e in tbl] == \
       [ (*map(int,fd.PFX_A[i]), int(fd.PFX_EPS[i]), int(fd.PFX_DELTA[i]), int(fd.PFX_D[i])) for i in range(fd.NPFX)]
print("table order/d OK")
random.seed(1)
nchk = 0
for f in (4, 8, 12):
    F, rows = parse_fits_file(U / "nick_test" / f"fits_f={f}.txt")
    _, th, g = rows[random.randrange(len(rows))]
    V = build_ring(g, F)
    for step in range(3):
        s = cr.sde_chi_full(V[0][0])
        n, ff = fd._column_numerators(V)
        new_s = fd._single_scan(n, ff)
        ref = np.array([cr.sde_chi_full(cr._prefix_times_V_00(e.P, V)) for e in tbl])
        assert (ref == new_s).all(), (f, step, np.nonzero(ref != new_s))
        new_s_o = fd._single_scan(n.astype(object), ff)
        assert (np.asarray(new_s_o, dtype=np.int64) == ref).all()
        assert (fd._single_scan_nb(n, ff) == ref).all()
        keys = fd._selection_keys()
        if step == 0:
            for ss in (s, s - 1):   # double search at the real s and a perturbed s
                ro = cr._try_double_prefix(V, ss, tbl)
                for nn in (n, n.astype(object)):
                    rf = fd._double_scan(nn, ff, ss, keys, new_s, fd._terms(nn))
                    ro_t = None if ro is None else (tbl.index(ro[0]), tbl.index(ro[1]), ro[2])
                    rf_t = None if rf is None else rf[:3]
                    assert ro_t == rf_t, (f, ss, ro_t, rf_t)
                rn = fd._double_scan_nb(n, ff, ss, keys, new_s)
                assert ro_t == (None if rn is None else rn[:3]), ("nb", f, ss, ro_t, rn)
                print("double OK", f, ss, ro_t, flush=True)
        nchk += 1
        idx = fd._pick_strict(new_s, s, fd._selection_keys())
        if idx < 0: break
        V = cr._reduce_by_three(cr._prefix_times_V(tbl[idx].P, V))
print("kernel checks OK", nchk)
