#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from typing import List

import numpy as np


PATTERN_RE = re.compile(r"^(.*)\.candidates_(\d{6})\.npz$")


def find_candidate_files(prefix: str = None, pattern: str = None) -> List[str]:
    if pattern is None:
        if prefix is None:
            raise ValueError("Either prefix or pattern must be provided")
        pattern = f"{prefix}.candidates_*.npz"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No candidate files matched: {pattern}")

    def sort_key(path: str):
        m = PATTERN_RE.match(os.path.basename(path))
        if m:
            return (m.group(1), int(m.group(2)))
        return (path, 0)

    return sorted(files, key=sort_key)


def merge_candidate_files(files: List[str], sort_by_dist: bool = True):
    dist_parts = []
    x_parts = []
    u_parts = []
    y_parts = []
    total = 0

    for i, path in enumerate(files, 1):
        with np.load(path) as data:
            required = ("dist", "x", "u_coeffs", "Y")
            missing = [k for k in required if k not in data]
            if missing:
                raise KeyError(f"{path} is missing arrays: {missing}")
            dist = np.array(data["dist"], copy=False)
            x = np.array(data["x"], copy=False)
            u = np.array(data["u_coeffs"], copy=False)
            y = np.array(data["Y"], copy=False)

        n = int(dist.shape[0])
        if x.shape[0] != n or u.shape[0] != n or y.shape[0] != n:
            raise ValueError(f"Inconsistent leading dimensions in {path}")

        dist_parts.append(dist.copy())
        x_parts.append(x.copy())
        u_parts.append(u.copy())
        y_parts.append(y.copy())
        total += n
        print(f"loaded {i}/{len(files)}: {path} ({n} candidates)")

    if total == 0:
        return {
            "dist": np.empty((0,), dtype=np.float64),
            "x": np.empty((0, 3), dtype=np.complex128),
            "u_coeffs": np.empty((0, 3, 6), dtype=np.int64),
            "Y": np.empty((0, 3, 3), dtype=np.int64),
        }

    dist = np.concatenate(dist_parts, axis=0)
    x = np.concatenate(x_parts, axis=0)
    u = np.concatenate(u_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if sort_by_dist and dist.shape[0] > 1:
        order = np.argsort(dist)
        dist = dist[order]
        x = x[order]
        u = u[order]
        y = y[order]

    return {
        "dist": dist,
        "x": x,
        "u_coeffs": u,
        "Y": y,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge find_roots candidate chunk .npz files into one .npz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix", help="Chunk file prefix, e.g. R2_f=4_eps=4e-5")
    group.add_argument("--pattern", help="Glob pattern, e.g. 'R2_f=4_eps=4e-5.candidates_*.npz'")
    parser.add_argument("--output", required=True, help="Output merged .npz file")
    parser.add_argument("--no_sort", action="store_true", help="Do not sort merged candidates by dist")
    args = parser.parse_args()

    files = find_candidate_files(prefix=args.prefix, pattern=args.pattern)
    print(f"found {len(files)} candidate files")
    merged = merge_candidate_files(files, sort_by_dist=not args.no_sort)
    np.savez(args.output, **merged)
    print(f"wrote {merged['dist'].shape[0]} merged candidates to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
