"""greedy_single=True / skip_double=True agreement on a few small-f matrices."""
import sys
from pathlib import Path
U = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parent), str(U / "nick_test"), str(U), str(U / "hrsa")]
import canonical_reducer as cr, fast_decompose as fd
from ingest_decompose import parse_fits_file, build_ring
def key(r):
    tc = [[(tuple(int(c) for c in e.num.coefs), e.denom_pow3) for e in row] for row in r["trailing_clifford"]]
    return (r["success"], r["D_count"], r["syllables"], tc, r.get("error"))
n = 0
for f in (4, 6, 8):
    F, rows = parse_fits_file(U / "nick_test" / f"fits_f={f}.txt")
    for _, th, g in rows[:: len(rows) // 3][:3]:
        for kw in (dict(greedy_single=True), dict(skip_double=True)):
            a = cr.decompose_canonical(build_ring(g, F), **kw)
            b = fd.decompose_fast(build_ring(g, F), **kw)
            assert key(a) == key(b), (f, th, kw)
            n += 1
print("greedy/skip_double agreement OK on", n, "runs")
