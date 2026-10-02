#!/usr/bin/env python3
"""sample_topk.py — stream a (gzipped) fits_topk_f=N.txt and keep every candidate
row for N_THETA evenly spaced theta values (sorted order), so that analyze_topk.py
can measure best-of-K on a manageable subset.

Usage: python3 sample_topk.py IN.txt.gz OUT.txt [--n-theta 30]
"""
import argparse
import gzip

import numpy as np


def opener(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--n-theta", type=int, default=30)
    a = ap.parse_args()
    thetas = set()
    with opener(a.inp) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            thetas.add(line.split(",", 1)[0].strip())
    ths = sorted(thetas, key=float)
    idx = sorted(set(np.linspace(0, len(ths) - 1, min(a.n_theta, len(ths))).round().astype(int)))
    keep = {ths[i] for i in idx}
    n = 0
    with opener(a.inp) as fh, open(a.out, "w") as out:
        for line in fh:
            if line.startswith("#"):
                out.write(line)
                continue
            if line.split(",", 1)[0].strip() in keep:
                out.write(line)
                n += 1
    print(f"{a.inp}: {len(ths)} thetas -> kept {len(keep)} thetas, {n} candidate rows -> {a.out}")


if __name__ == "__main__":
    main()
