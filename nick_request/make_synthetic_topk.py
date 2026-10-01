"""Build a synthetic top-K file from Nick's existing fits_f=N.txt (test input for
analyze_topk.py): for every theta in the file, rank ALL matrices of the file by
||M - R_z(theta)||_F, keep those with frob <= tol*best (at most K), and write them
verbatim in the top-K format (fits rows + ', frob , rank').  This is exactly what
dump_special_thetas.py --topk does on Nick's side, but with the smaller pool he sent us."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nick_test"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_topk import parse_topk_file
from ingest_decompose import build_complex
ap = argparse.ArgumentParser()
ap.add_argument("fits", type=Path); ap.add_argument("--k", type=int, default=30)
ap.add_argument("--tol", type=float, default=1.25); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
f, rows = parse_topk_file(a.fits)
Ms = np.array([build_complex(r["groups"], f) for r in rows])
raw = [l.rstrip("\n") for l in open(a.fits) if l.strip() and not l.startswith("#")]
ths = sorted({r["theta"] for r in rows})
n = 0; sizes = []
with open(a.out, "w") as fh:
    fh.write(f"#f={f}\n# columns: theta, 3^f*Mij (3x3 6-element-groups), frob, rank\n")
    for th in ths:
        T = np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1.0])
        fr = np.linalg.norm(Ms - T, axis=(1, 2))
        o = np.argsort(fr, kind="mergesort")
        keep = [i for i in o if fr[i] <= a.tol * fr[o[0]]][: a.k]
        seen = set(); r = 0
        for i in keep:
            body = raw[i].split(",", 1)[1]
            if body in seen: continue
            seen.add(body)
            fh.write(f"{th} ,{body} , {fr[i]:.17g} , {r}\n"); r += 1; n += 1
        sizes.append(r)
print(f"f={f}: {len(ths)} thetas, {n} rows, candidates/theta mean {np.mean(sizes):.2f} max {max(sizes)} -> {a.out}")
