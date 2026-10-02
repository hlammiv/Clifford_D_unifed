"""Profile the original canonical_reducer.decompose_canonical (task 1)."""
import cProfile, pstats, sys, time, io
from pathlib import Path
U = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(U / "nick_test"), str(U), str(U / "hrsa")]
import canonical_reducer as cr
from ingest_decompose import parse_fits_file, build_ring

t0 = time.time(); cr.get_prefix_table(); print(f"table build {time.time()-t0:.1f}s", flush=True)
# instrument double-prefix calls
orig_dbl = cr._try_double_prefix
stats = {"dbl_calls": 0, "dbl_time": 0.0}
def dbl(*a, **k):
    t = time.time(); r = orig_dbl(*a, **k)
    stats["dbl_calls"] += 1; stats["dbl_time"] += time.time() - t; return r
cr._try_double_prefix = dbl
for f in map(int, sys.argv[1].split(",")):
    F, rows = parse_fits_file(U / "nick_test" / f"fits_f={f}.txt")
    for i, (_, th, g) in enumerate(rows[:3]):
        stats.update(dbl_calls=0, dbl_time=0.0)
        V = build_ring(g, F)
        if i == 0:
            pr = cProfile.Profile(); t = time.time(); pr.enable()
            r = cr.decompose_canonical(V); pr.disable(); w = time.time() - t
            s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
            print(s.getvalue())
        else:
            t = time.time(); r = cr.decompose_canonical(V); w = time.time() - t
        print(f"f={F} row{i} wall={w:.1f}s{' (cProfile)' if i==0 else ''} s0={r['sde_chi_initial']} "
              f"n_iter={r['n_iter']} nsyl={len(r['syllables'])} N_D={r['D_count']} "
              f"dbl_calls={stats['dbl_calls']} dbl_time={stats['dbl_time']:.1f}s", flush=True)
