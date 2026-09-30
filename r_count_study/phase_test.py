"""phase_test.py — does a global phase (V, -V, zeta^k V) change the R-count from canonical_reducer?
Also prints det of the gate generators (sign bookkeeping for the R-parity invariant)."""
import sys, random
from pathlib import Path
import numpy as np
import analyze_candidates as A

cr = A.cr
print("det H =", np.linalg.det(A.Vc(cr.gate_H())), " sign:", A.unit_sign(np.linalg.det(A.Vc(cr.gate_H()))))
logs = sys.argv[1:] or [str(p) for p in sorted(Path(__file__).parent.glob("raw/hrsa_t*_e0.01.log"))]
cr.get_prefix_table()
for lg in logs[:3]:
    it = A.parse_log(Path(lg))
    for x in random.Random(7).sample(it, 2):
        V = A.build_V(x[4], x[5], x[6], x[3])
        out = []
        for lab, ph in [("V", cr._zeta9_power(0))] + [(f"z^{k}V", cr._zeta9_power(k)) for k in (1, 3)] + \
                       [("-V", -cr._zeta9_power(0)), ("-z^1V", -cr._zeta9_power(1))]:
            W = A.scale(V, ph)
            r = cr.decompose_canonical(W)
            nR, *_ = A.analyze_syllables(r)
            signs, rR, _ = A.trailing_info(r["trailing_clifford"])
            same = [(s["a0"], s["a1"], s["a2"], s["eps"], s["delta"]) for s in r["syllables"]]
            out.append((lab, r["D_count"], nR, "".join("+" if s > 0 else "-" for s in signs), rR, hash(tuple(same)) % 10000,
                        A.det_sign(W)))
        print(x[0], out, flush=True)
