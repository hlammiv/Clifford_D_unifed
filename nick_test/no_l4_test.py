"""no_l4_test.py — can Nick's matrices be decomposed using only level-3 (T-type)
or Clifford syllable diagonals, i.e. (a1+a2+a3) % 3 == 0 (a0 = 0 in the table)?
Variants: with R allowed / without R. Repo code unchanged (table filtered in-process)."""
import sys, argparse
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent), str(_HERE.parent / "hrsa")]
import canonical_reducer as cr
from ingest_decompose import parse_fits_file, build_ring
ap = argparse.ArgumentParser(); ap.add_argument("--f", type=int, default=10); ap.add_argument("--n", type=int, default=10)
a = ap.parse_args()
f, rows = parse_fits_file(_HERE / f"fits_f={a.f}.txt")
full = cr.get_prefix_table()
print("table entry sample fields:", [(e.a1, e.a2, e.a3) for e in full[:3]])
variants = {
    "all (baseline)": full,
    "level<=3 diag, R allowed": [e for e in full if (e.a1 + e.a2 + e.a3) % 3 == 0],
    "level<=3 diag, no R": [e for e in full if (e.a1 + e.a2 + e.a3) % 3 == 0 and e.eps == 0],
}
for label, tbl in variants.items():
    cr._PREFIX_TABLE_CACHE = tbl
    ok = 0; nd = []; fails = []
    for _, th, g in rows[: a.n]:
        r = cr.decompose_canonical(build_ring(g, f))
        if r["success"]: ok += 1; nd.append(r["D_count"])
        else: fails.append(r.get("sde_chi_final"))
    print(f"{label:28s} table={len(tbl):5d}  success {ok}/{a.n}  mean N_D {sum(nd)/max(len(nd),1):.1f}  stalled at sde {fails}", flush=True)
