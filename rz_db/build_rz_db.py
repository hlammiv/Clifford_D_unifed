#!/usr/bin/env python3
"""build_rz_db.py — Ingest sweep CSV(s) + per-cell JSONs into an rz_lookup SQLite DB.

CSV schemas supported (detected by header):
  * zeta9 calibration ("max_f, theta_idx, theta, eps_target, ...")
       JSON file path = <csv_dir>/zeta9_maxf{max_f}_t{theta_idx:03d}.json
  * HRSA grid / sweep_hrsa.py ("theta_idx, theta, epsilon, method, ...")
       JSON file path = <csv_dir>/hrsa_t{theta:.6f}_eps{epsilon:g}.json
       (dots replaced by underscores, matching sweep_hrsa_grid.py)
  * HRSA legacy (no theta_idx, no per-cell JSON) — rows are warned and skipped.

Usage:
    build_rz_db.py [--db PATH] [--rebuild] [--verbose] SWEEP_CSV [SWEEP_CSV ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script: `python rz_db/build_rz_db.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rz_db.rz_lookup import RzLookupDB, V_SHAPE  # noqa: E402


def detect_schema(fieldnames: list[str]) -> str:
    fs = set(fieldnames or [])
    if "max_f" in fs and "theta_idx" in fs and "eps_target" in fs:
        return "zeta9_cal"
    if "theta_idx" in fs and "epsilon" in fs and "method" in fs:
        return "hrsa_grid"
    if "theta" in fs and "epsilon" in fs and "method" in fs:
        return "hrsa_legacy"
    return "unknown"


def hrsa_label(theta: float, eps: float) -> str:
    """Replicate sweep_hrsa_grid.py naming: t{theta:.6f}_eps{eps:g}, dots->underscores."""
    return f"t{theta:.6f}_eps{eps:g}".replace(".", "_")


def parse_row_zeta9_cal(row: dict, csv_dir: Path):
    """Return (json_path, eps_target, source_tag) for a zeta9 calibration row."""
    max_f = int(row["max_f"])
    theta_idx = int(row["theta_idx"])
    json_path = csv_dir / f"zeta9_maxf{max_f}_t{theta_idx:03d}.json"
    eps_target = float(row["eps_target"])
    source = f"zeta9_cal:maxf{max_f}:t{theta_idx:03d}"
    return json_path, eps_target, source


def parse_row_hrsa_grid(row: dict, csv_dir: Path):
    """Return (json_path, eps_target, source_tag) for an HRSA grid row."""
    theta = float(row["theta"])
    eps_target = float(row["epsilon"])
    label = hrsa_label(theta, eps_target)
    json_path = csv_dir / f"hrsa_{label}.json"
    theta_idx = row.get("theta_idx") or ""
    source = f"hrsa_grid:t{theta_idx}:eps{eps_target:g}" if theta_idx else f"hrsa_grid:eps{eps_target:g}"
    return json_path, eps_target, source


def is_success_zeta9_cal(row: dict) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def is_success_hrsa(row: dict) -> bool:
    # all_checks_pass may have trailing \r; strip it
    return str(row.get("all_checks_pass", "")).strip().lower() == "true"


def load_cell_json(json_path: Path) -> dict | None:
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def extract_entry(cell: dict, eps_target: float, source: str) -> dict | None:
    """Turn a per-cell schema dict into kwargs for RzLookupDB.insert.

    Returns None if the cell is unusable (missing unitary, bad shape, etc).
    """
    unitary = cell.get("unitary")
    if unitary is None:
        return None
    V_nested = unitary.get("V")
    if V_nested is None:
        return None
    try:
        V = np.array(V_nested, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if V.shape != V_SHAPE:
        return None
    v_f = unitary.get("f")
    if v_f is None:
        return None

    achieved = cell.get("achieved") or {}
    inputs = cell.get("inputs") or {}
    decomp = cell.get("decomposition") or {}

    theta = inputs.get("theta")
    if theta is None:
        return None

    achieved_frob = achieved.get("achieved_frob")
    if achieved_frob is None:
        return None

    # N_D may legitimately be 0 (SignExtClifford / Direct). Treat -1 / negative as
    # "failed" sentinel and skip the entry entirely.
    N_D_raw = decomp.get("N_D")
    if N_D_raw is None:
        N_D = None
    else:
        try:
            n = int(N_D_raw)
        except (TypeError, ValueError):
            N_D = None
        else:
            if n < 0:
                return None
            N_D = n

    method = achieved.get("method") or "unknown"

    return {
        "theta": float(theta),
        "eps_target": float(eps_target),
        "V": V,
        "v_f": int(v_f),
        "achieved_frob": float(achieved_frob),
        "N_D": N_D,
        "method": str(method),
        "source": source,
    }


def ingest_csv(csv_path: Path, db: RzLookupDB, verbose: bool = False) -> dict:
    """Walk one CSV, insert successful cells, return per-csv counters."""
    counters = {
        "csv": str(csv_path),
        "rows_total": 0,
        "rows_not_success": 0,
        "rows_inserted": 0,
        "rows_skipped_missing_json": 0,
        "rows_skipped_parse_error": 0,
        "rows_skipped_bad_unitary": 0,
        "schema": None,
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from field names (handles \r etc.)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
        schema = detect_schema(reader.fieldnames or [])
        counters["schema"] = schema
        csv_dir = csv_path.parent

        if schema == "unknown":
            print(f"  [skip] {csv_path.name}: unknown schema (fields: {reader.fieldnames})",
                  file=sys.stderr)
            return counters

        if schema == "hrsa_legacy":
            print(f"  [warn] {csv_path.name}: legacy HRSA schema has no per-cell JSONs; "
                  f"all rows will be skipped.", file=sys.stderr)
            for row in reader:
                counters["rows_total"] += 1
                counters["rows_skipped_missing_json"] += 1
            return counters

        for row in reader:
            counters["rows_total"] += 1

            # Strip CRs from all values (CSVs may be CRLF)
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

            if schema == "zeta9_cal":
                if not is_success_zeta9_cal(row):
                    counters["rows_not_success"] += 1
                    continue
                try:
                    json_path, eps_target, source = parse_row_zeta9_cal(row, csv_dir)
                except (KeyError, ValueError):
                    counters["rows_skipped_parse_error"] += 1
                    continue
            elif schema == "hrsa_grid":
                if not is_success_hrsa(row):
                    counters["rows_not_success"] += 1
                    continue
                try:
                    json_path, eps_target, source = parse_row_hrsa_grid(row, csv_dir)
                except (KeyError, ValueError):
                    counters["rows_skipped_parse_error"] += 1
                    continue
            else:
                continue

            cell = load_cell_json(json_path)
            if cell is None:
                counters["rows_skipped_missing_json"] += 1
                if verbose:
                    print(f"    [miss] {json_path.name}", file=sys.stderr)
                continue

            entry = extract_entry(cell, eps_target, source)
            if entry is None:
                counters["rows_skipped_bad_unitary"] += 1
                if verbose:
                    print(f"    [bad-V] {json_path.name}", file=sys.stderr)
                continue

            db.insert(**entry)
            counters["rows_inserted"] += 1

    db.commit()
    return counters


def main():
    p = argparse.ArgumentParser(description="Ingest sweep CSVs into an rz_lookup SQLite DB.")
    p.add_argument("--db", default="rz_lookup.sqlite",
                   help="Path to the SQLite DB file (default: ./rz_lookup.sqlite).")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete the DB before ingesting.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print each skipped/missing cell.")
    p.add_argument("csvs", nargs="+", help="One or more sweep summary CSV files.")
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    if args.rebuild and db_path.exists():
        print(f"[rz_db] removing existing DB {db_path}", file=sys.stderr)
        db_path.unlink()

    db = RzLookupDB(str(db_path))
    print(f"[rz_db] DB = {db_path}  (existing entries: {db.count()})", file=sys.stderr)

    grand_total = 0
    for csv_arg in args.csvs:
        csv_path = Path(csv_arg).resolve()
        if not csv_path.exists():
            print(f"  [skip] {csv_path}: not found", file=sys.stderr)
            continue
        print(f"[rz_db] ingesting {csv_path}", file=sys.stderr)
        c = ingest_csv(csv_path, db, verbose=args.verbose)
        print(
            f"  schema={c['schema']}  rows={c['rows_total']}  inserted={c['rows_inserted']}  "
            f"not_success={c['rows_not_success']}  missing_json={c['rows_skipped_missing_json']}  "
            f"bad_unitary={c['rows_skipped_bad_unitary']}  parse_err={c['rows_skipped_parse_error']}",
            file=sys.stderr,
        )
        grand_total += c["rows_inserted"]

    db.commit()
    final = db.count()
    print(f"[rz_db] done. inserted_this_run={grand_total}  total_in_db={final}", file=sys.stderr)
    # Brief stats summary on stdout
    stats = db.stats()
    print(json.dumps({
        "db_path": str(db_path),
        "total": stats["total"],
        "eps_tiers": {f"{k:g}": v["n"] for k, v in stats["eps_tiers"].items()},
        "by_source_prefix": {
            prefix: sum(v for k, v in stats["by_source"].items() if k.startswith(prefix))
            for prefix in ("zeta9_cal", "hrsa_grid")
        },
    }, indent=2))
    db.close()


if __name__ == "__main__":
    main()
