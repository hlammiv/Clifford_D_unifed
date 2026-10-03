"""plot_nd_vs_theta.py — Nick's decomposed D-gate count N_D as a function of θ,
one series per f. This is the fixed-f gate-count slice: ε is ~flat per f, so the
synthesis variance shows up HERE (the C+R / C+T 'T-count variance' analogue).
Source: nick_nd_vs_eps_all.csv (theta, N_D, method=nick(f=N))."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
CSV = _HERE / "nick_nd_vs_eps_all.csv"
COLORS = {4: "#ffbb33", 6: "#ff7f0e", 8: "#e6550d",
          10: "#d62728", 12: "#8c2d04", 14: "#9467bd", 16: "#4a1486"}

by_f = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    f = int(r["method"].split("=")[1].rstrip(")"))
    by_f[f].append((float(r["theta"]), int(r["N_D"])))

fig, ax = plt.subplots(figsize=(10, 6))
for f in sorted(by_f):
    pts = np.array(sorted(by_f[f]))
    th, nd = pts[:, 0], pts[:, 1]
    ax.plot(th, nd, ".", ms=4, color=COLORS[f], alpha=0.7,
            label=f"f={f}  (n={len(nd)}, N_D {int(nd.min())}–{int(nd.max())}, "
                  f"med {int(np.median(nd))}, σ={nd.std():.1f})")

ax.set_xlabel(r"$\theta$  (target $R^Z(\theta)=\mathrm{diag}(e^{-i\theta/2},e^{+i\theta/2},1)$)")
ax.set_ylabel(r"$N_D$  (non-Clifford D-gate count)")
ax.set_title("Nick exact-ring Clifford+D: gate count $N_D$ vs angle, by level f\n"
             "(fixed-f slice: ε is flat, so synthesis variance lives in $N_D$)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.93, ncol=1)
fig.tight_layout()
out = _HERE.parent / "nick_nd_vs_theta_2026-05-29.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
for f in sorted(by_f):
    nd = np.array([n for _, n in by_f[f]])
    print(f"  f={f:2d}: N_D mean={nd.mean():.1f} std={nd.std():.1f} "
          f"CV={nd.std()/nd.mean():.3f} range={nd.min()}-{nd.max()}")
