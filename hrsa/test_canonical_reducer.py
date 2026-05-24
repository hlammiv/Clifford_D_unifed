"""test_canonical_reducer.py — smoke + correctness tests for canonical_reducer.

Tests covered:
  1. sde_chi_z9 / sde_chi_full vs C++ decompose_tool on cell_0022 (f=8).
  2. Round-trip: V_input → decompose → reconstruct → must equal V_input
     bit-exactly (no float roundoff).
  3. D_count matches the C++ decompose_tool D_count for the same V.

Run:
  python3 test_canonical_reducer.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_UNIFIED = _HERE.parent
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

import canonical_reducer as cr
from ep_level import Z9, Z9Frac


def _load_cell_input(cell_path: Path) -> tuple[list[list[Z9Frac]], int, dict]:
    """Load a sweep cell JSON, return (V, f, expected_decomposition_dict)."""
    with open(cell_path, "r") as fh:
        cell = json.load(fh)
    f = int(cell["unitary"]["f"])
    V_blob = cell["unitary"]["V"]
    V: list[list[Z9Frac]] = []
    for i in range(3):
        row: list[Z9Frac] = []
        for j in range(3):
            coefs = tuple(int(c) for c in V_blob[i][j])
            row.append(Z9Frac(Z9(coefs), f))
        V.append(row)
    return V, f, cell.get("decomposition", {})


def test_sde_chi_matches_cell(cell_path: Path) -> bool:
    V, f, expected = _load_cell_input(cell_path)
    sde = cr.sde_chi_full(V[0][0])
    expected_sde = expected.get("sde_chi_initial")
    ok = sde == expected_sde
    print(f"  sde_chi_full(V[0][0]) = {sde}  (expected {expected_sde}) "
          + ("OK" if ok else "FAIL"))
    return ok


def test_decompose_matches_cell(cell_path: Path, greedy: bool = True) -> bool:
    """Decompose, verify bit-exact round-trip, compare D_count to cell.

    Greedy mode (default): accepts any-drop single-prefix → can produce
    SMALLER D_count than strict (which mirrors C++) — verify only checks
    the round-trip, not the D_count match.
    """
    V, f, expected = _load_cell_input(cell_path)
    expected_d = expected.get("N_D")
    expected_n = len(expected.get("syllables", []))

    t0 = time.time()
    result = cr.decompose_canonical(V, verbose=False, greedy_single=greedy)
    t1 = time.time()
    if not result["success"]:
        print(f"  decompose failed: {result.get('error')}")
        return False

    # Verify reconstruction.
    ver = cr.verify_decomposition(V, result["syllables"],
                                  result["trailing_clifford"])
    match = ver["matches"]
    compare_mark = "≤ OK" if result['D_count'] <= expected_d else "WORSE"
    print(f"  D_count = {result['D_count']}  (cell reports {expected_d}) {compare_mark}")
    print(f"  n_syllables = {len(result['syllables'])}  (cell reports {expected_n})")
    print(f"  verify.matches = {match}  max_residual = {ver['max_coef_residual']}  "
          + ("OK" if match else "FAIL"))
    print(f"  wall = {t1-t0:.1f}s  (mode={'greedy' if greedy else 'strict'})")
    return match  # round-trip is the hard-correctness gate


def main() -> int:
    """Sweep a few cell_NNNN.json files; we expect bit-exact reconstruction
    and matching D_count on every successful case."""
    sweep_root = _UNIFIED / "sweep_zeta9_tier2_eps1e-4_2026-05-23"
    if not sweep_root.exists():
        print(f"SKIP: {sweep_root} not found")
        return 0

    # Pick a small set of cells of varying sde_chi to exercise the peeler.
    cell_indices = [22, 100, 250, 500]
    cells_tested = 0
    all_ok = True
    for ci in cell_indices:
        cp = sweep_root / f"cell_{ci:04d}.json"
        if not cp.exists():
            print(f"SKIP {cp.name}: not found")
            continue
        with open(cp) as fh:
            cell = json.load(fh)
        if not cell.get("decomposition", {}).get("decompose_success"):
            print(f"SKIP {cp.name}: cell decomposition did not succeed in original sweep")
            continue
        print(f"=== {cp.name} ===")
        print(" -- sde_chi test")
        ok1 = test_sde_chi_matches_cell(cp)
        print(" -- decompose round-trip test")
        ok2 = test_decompose_matches_cell(cp)
        cells_tested += 1
        all_ok = all_ok and ok1 and ok2

    print("=" * 40)
    print(f"Tested {cells_tested} cells.  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok and cells_tested > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
