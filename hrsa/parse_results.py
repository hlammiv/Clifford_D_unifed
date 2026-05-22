#!/usr/bin/env python3
"""
Parse output files of the form:
    out_a<angle>_e<epsilon>_f<something>

Extracts from each file:
  - angle        : number after 'a' in filename
  - target_eps   : number after 'e' in filename
  - D_gates      : the "Total D gates" value from the decomposition
  - eps_diff_val : the Matrix Frobenius distance (preferred) or
                   the "Epsilon Diff. Val." (fallback)
  - result       : "Success" or "Failure"

Usage:
    python parse_results.py [data_dir] [--output results.csv]

    data_dir defaults to ./data
"""

import os
import re
import csv
import sys
import argparse


# ── filename parsing ──────────────────────────────────────────────────────────

FNAME_RE = re.compile(
    r"out_a(?P<angle>-?[0-9]+(?:\.[0-9]+)?)_e(?P<target_eps>-?[0-9]+(?:\.[0-9]+)?)_f"
)

def parse_filename(name):
    """Return (angle, target_eps) strings from the base filename, or None."""
    m = FNAME_RE.search(os.path.basename(name))
    if not m:
        return None, None
    return m.group("angle"), m.group("target_eps")


# ── file-content parsing ──────────────────────────────────────────────────────

D_GATES_RE   = re.compile(r"Total D gates\s*:\s*(\d+)")
FROB_RE      = re.compile(
    r"Matrix Frobenius distance.*?=\s*([0-9eE+\-.]+)"
)
EPS_DIFF_RE  = re.compile(r"Epsilon Diff\. Val\.\s*:\s*([0-9eE+\-.]+)")
SUCCESS_RE   = re.compile(r"\bSuccess\b", re.IGNORECASE)
FAILURE_RE   = re.compile(r"\bFailure\b", re.IGNORECASE)

def parse_file(path):
    """
    Returns a dict with keys: D_gates, eps_diff_val, result
    Any field that cannot be found is set to None.
    """
    d_gates     = None
    frob_dist   = None
    eps_diff    = None
    result      = None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            # D-gate count from decomposition
            m = D_GATES_RE.search(line)
            if m:
                d_gates = int(m.group(1))

            # Matrix Frobenius distance (preferred measure)
            m = FROB_RE.search(line)
            if m:
                frob_dist = float(m.group(1))

            # Epsilon diff value (fallback)
            m = EPS_DIFF_RE.search(line)
            if m:
                eps_diff = float(m.group(1))

            # success / failure
            if SUCCESS_RE.search(line):
                result = "Success"
            elif FAILURE_RE.search(line):
                result = "Failure"

    # prefer matrix Frobenius distance over vector-level eps_diff
    best_eps = frob_dist if frob_dist is not None else eps_diff

    return {
        "D_gates":      d_gates,
        "eps_diff_val": best_eps,
        "result":       result,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse simulation output files.")
    parser.add_argument("data_dir", nargs="?", default="data",
                        help="Directory containing output files (default: ./data)")
    parser.add_argument("--output", default="results.csv",
                        help="CSV output path (default: results.csv)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = []

    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue

        angle, target_eps = parse_filename(fname)
        if angle is None:
            skipped.append(fname)
            continue

        info = parse_file(fpath)
        rows.append({
            "filename":     fname,
            "angle":        angle,
            "target_eps":   target_eps,
            "D_gates":      info["D_gates"],
            "eps_diff_val": info["eps_diff_val"],
            "result":       info["result"],
        })

    # write space-delimited file
    fieldnames = ["angle", "target_eps", "D_gates", "eps_diff_val", "result"]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=" ", extrasaction="ignore")
        csvfile.write("# ")
        writer.writeheader()
        for row in rows:
            if any(row[f] is None for f in fieldnames):
                missing = [f for f in fieldnames if row[f] is None]
                csvfile.write(f"# error - missing: {', '.join(missing)} | "
                              f"angle={row['angle']} target_eps={row['target_eps']} "
                              f"D_gates={row['D_gates']} eps_diff_val={row['eps_diff_val']} "
                              f"result={row['result']}\n")
            else:
                writer.writerow(row)

    print(f"Parsed {len(rows)} file(s) → {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with unrecognised names: {skipped}")


if __name__ == "__main__":
    main()
