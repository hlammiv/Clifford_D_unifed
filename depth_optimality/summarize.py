"""summarize.py — per-f means and log3 fits of T-count / T-depth columns from results*.jsonl."""
import glob, json, math
import numpy as np
rows = [json.loads(l) for fn in sorted(glob.glob("results*.jsonl")) for l in open(fn)]
cols = ["n_syl", "T_shared", "T_merged", "wire_shared", "rot_shared", "wire_fresh", "rot_fresh",
        "T_meas", "wire_meas", "rot_meas"]
fs = sorted({r["f"] for r in rows})
print("f   n  log3(1/eps) " + " ".join(f"{c:>11s}" for c in cols))
for f in fs:
    s = [r for r in rows if r["f"] == f]
    print(f"{f:<3d} {len(s):<2d} {np.mean([math.log(1/r['eps'], 3) for r in s]):11.2f} "
          + " ".join(f"{np.mean([r[c] for r in s]):11.1f}" for c in cols))
L = np.array([math.log(1 / r["eps"], 3) for r in rows]); A = np.vstack([np.ones_like(L), L]).T
print("fits a + b*log3(1/eps), n=%d:" % len(rows))
for c in cols:
    a, b = np.linalg.lstsq(A, np.array([r[c] for r in rows], float), rcond=None)[0]
    print(f"  {c:12s} {a:6.2f} + {b:5.2f}*log3")
print("max merge_err %.1e  max sim_err_meas %.1e" % (max(r["merge_err"] for r in rows), max(r["sim_err_meas"] for r in rows)))
print("T_shared == 1*T3+7*L4+7*R:", all(r["T_shared"] == r["n_T3"] + 7 * r["n_L4"] + 7 * r["n_R"] for r in rows),
      " T_meas == T3+4*(L4+R):", all(r["T_meas"] == r["n_T3"] + 4 * r["n_L4"] + 4 * r["n_R"] for r in rows))
