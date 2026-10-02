import cProfile, pstats, sys
from pathlib import Path
U = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parent), str(U / "nick_test"), str(U), str(U / "hrsa")]
import fast_decompose as fd
from ingest_decompose import parse_fits_file, build_ring
F, rows = parse_fits_file(U / "nick_test" / f"fits_f={sys.argv[1]}.txt")
V = build_ring(rows[0][2], F)
pr = cProfile.Profile(); pr.enable(); r = fd.decompose_fast(V); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
