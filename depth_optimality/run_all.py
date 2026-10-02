"""run_all.py — T-count / T-depth table over sample matrices at given f values."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from emit_variants import emit, simulate_branch, cc, parse_fits_file, build_ring, U_
from tdepth import wire_depth, rot_depth, tcount
from tmerge import merge, merge_symbolic
ap = argparse.ArgumentParser(); ap.add_argument("--f", type=int, nargs="+", default=[4, 8])
ap.add_argument("--n", type=int, default=10); ap.add_argument("--out", default=str(HERE / "results.jsonl"))
a = ap.parse_args()
rng = np.random.default_rng(0)
fo = open(a.out, "a")
for f in a.f:
    _, rows = parse_fits_file(U_ / "nick_test" / f"fits_f={f}.txt")
    for i in np.linspace(0, len(rows) - 1, a.n).astype(int):
        _, th, g = rows[i]
        V = build_ring(g, f); Vc = cc.ring_to_complex(V)
        eps = float(np.linalg.norm(Vc - np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1])))
        cc.cr.set_selection_cost("tcost")
        r = cc.cr.decompose_canonical(V)
        row = dict(f=f, theta=th, eps=eps, n_syl=len(r["syllables"]))
        for mode in ("shared", "fresh", "meas"):
            gl, cnt = emit(r["syllables"], r["trailing_clifford"], mode, rng)
            row.update({k: v for k, v in cnt.items()})
            row[f"T_{mode}"] = tcount(gl)
            row[f"wire_{mode}"] = wire_depth([x for x in gl if x.kind != "F"])
            row[f"rot_{mode}"] = rot_depth(gl)
            row[f"Tmerge_sym_{mode}"] = merge_symbolic(gl)
            if mode == "shared":
                row.update(merge(gl))
            if mode == "meas":
                row["sim_err_meas"] = simulate_branch(gl, Vc, rng, 4)
                row["n_anc_meas"] = len({q for x in gl for q in x.qubits}) - 1
        print(json.dumps(row), flush=True); fo.write(json.dumps(row) + "\n"); fo.flush()
