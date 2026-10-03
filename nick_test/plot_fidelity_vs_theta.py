"""plot_fidelity_vs_theta.py — Nick's achieved approximation error (Frobenius
distance ε to the target R^Z(θ)=diag(e^{-iθ/2},e^{+iθ/2},1)) as a function of θ,
one series per f. Uses every angle in each fits_f=N.txt (no decomposition needed)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import ingest_decompose as ig

FILES = [("fits_f=4.txt", 4), ("fits_f=6.txt", 6), ("fits_f=8.txt", 8),
         ("fits_f=10.txt", 10), ("fits_f=12.txt", 12), ("fits_f=14.txt", 14),
         ("fits_f=16.txt", 16)]
COLORS = {4: "#ffbb33", 6: "#ff7f0e", 8: "#e6550d",
          10: "#d62728", 12: "#8c2d04", 14: "#9467bd", 16: "#4a1486"}

fig, ax = plt.subplots(figsize=(10, 6))
for fn, f in FILES:
    f_hdr, rows = ig.parse_fits_file(_HERE / fn)
    th = np.array([r[1] for r in rows])
    eps = np.array([ig.achieved_eps(ig.build_complex(r[2], f_hdr), r[1])[0]
                    for r in rows])
    order = np.argsort(th)
    ax.plot(th[order], eps[order], ".", ms=3, color=COLORS[f], alpha=0.7,
            label=f"f={f}  (n={len(rows)}, ε≈{np.median(eps):.1e})")

ax.set_yscale("log")
ax.set_xlabel(r"$\theta$  (target $R^Z(\theta)=\mathrm{diag}(e^{-i\theta/2},e^{+i\theta/2},1)$)")
ax.set_ylabel(r"achieved Frobenius distance  $\varepsilon = \|V-R^Z(\theta)\|_F$")
ax.set_title("Nick exact-ring Clifford+D: achieved error vs angle, by denominator level f")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=9, loc="center right", framealpha=0.93, title="level (lower ε = better)")
fig.tight_layout()
out = _HERE.parent / "nick_fidelity_vs_theta_2026-05-29.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
