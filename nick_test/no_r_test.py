"""no_r_test.py — can Nick's matrices be exactly decomposed WITHOUT R syllables?

Restricts canonical_reducer's prefix table to eps=0 (H·D·X^delta only) and
re-decomposes. Reports success, N_D, number of peel steps, and whether the
trailing monomial residual still carries an R (mono_r). Compare with the
unrestricted run. Repo code is not modified (table filtered in-process)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent), str(_HERE.parent / "hrsa")]
import canonical_reducer as cr          # noqa: E402
from ingest_decompose import parse_fits_file, build_ring  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--f", type=int, default=10)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--greedy", action="store_true")
a = ap.parse_args()
f, rows = parse_fits_file(_HERE / f"fits_f={a.f}.txt")
full = cr.get_prefix_table()
nor = [e for e in full if e.eps == 0]
for label, tbl in (("with R", full), ("no R", nor)):
    cr._PREFIX_TABLE_CACHE = tbl
    ok = 0; nd = []; rres = 0; rsyl = 0
    for _, th, g in rows[: a.n]:
        res = cr.decompose_canonical(build_ring(g, f), greedy_single=a.greedy)
        if res["success"]:
            ok += 1; nd.append(res["D_count"])
            rsyl += sum(s["eps"] for s in res["syllables"])
            _, _, mr = cr.classify_monomial_and_d_cost(res["trailing_clifford"])
            rres += bool(mr)
        else:
            print(f"  [{label}] theta={th:.4f} FAILED: {res.get('error')}")
    print(f"{label:7s}: success {ok}/{a.n}  mean N_D {sum(nd)/max(len(nd),1):.1f}  "
          f"R syllables total {rsyl}  residuals needing R {rres}", flush=True)
