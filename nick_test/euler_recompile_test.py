"""euler_recompile_test.py — does canonical reduction of a multi-axis R_z
product save D-gates over the naive sum-of-pieces upper bound?

Setup: Nick's fits_f=10.txt gives us R_z^(0,1)(theta) exact-ring matrices at
N_D=74 average. The qutrit cyclic shift X = [[0,0,1],[1,0,0],[0,1,0]] is a
Clifford, and conjugation X · R_z^(0,1)(theta) · X^{-1} = R_z^(1,2)(theta)
(verified by hand: X cycles 0->1, 1->2, 2->0). So we can build R_z^(1,2)
matrices for free from Nick's R_z^(0,1) by Clifford conjugation.

For each trial: pick K random Nick rows, alternate (0,1) and (1,2) subspaces,
multiply the ring matrices, then run canonical_reducer on the product. The
multi-axis product lives in a 4-D subgroup of SU(3) — non-trivial U(3) target
that activates two non-commuting Z-rotations.

Compare:
  N_D_naive   = sum of per-piece N_D from Nick's decompositions (upper bound)
  N_D_reduced = canonical_reducer(composed product)

ratio < 0.7  -> smoking gun for C+D win via canonical-form merging
ratio 0.85-0.95 -> real but modest, worth a paragraph
ratio ~ 1.0  -> dead idea, matches naive Euler scaling
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "hrsa"))
sys.path.insert(0, str(_HERE))

import canonical_reducer as cr  # noqa: E402
import ingest_decompose as ig   # noqa: E402

# Qutrit cyclic shift X: |j> -> |j+1 mod 3>
# X|0>=|1>, X|1>=|2>, X|2>=|0>  ==>  X = [[0,0,1],[1,0,0],[0,1,0]]
X_INT     = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
X_INV_INT = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


def make_int_ring_matrix(int_mat):
    """Build a 3x3 ring matrix with integer entries at denominator 1."""
    V = []
    for i in range(3):
        row = []
        for j in range(3):
            n = int_mat[i][j]
            coefs = (cr._INT(n), cr._INT(0), cr._INT(0),
                     cr._INT(0), cr._INT(0), cr._INT(0))
            row.append(cr._FastZ9Frac(cr._FastZ9(coefs), 0))
        V.append(row)
    return V


def conjugate_by_X(V):
    """X * V * X^{-1}  (Clifford conjugation, no D-gate cost)."""
    X = make_int_ring_matrix(X_INT)
    Xi = make_int_ring_matrix(X_INV_INT)
    return cr._fast_matmul(cr._fast_matmul(X, V), Xi)


def load_nick_pieces(fits_path, json_path, f_target):
    """Return list of dicts {row, theta, V_ring (R_z^(0,1) at level f), N_D_naive}.
    Pulls ring matrices from the .txt file and N_D from the decomposition JSON.
    """
    f_hdr, rows = ig.parse_fits_file(fits_path)
    assert f_hdr == f_target, f"expected f={f_target}, file has f={f_hdr}"
    with open(json_path) as fh:
        data = json.load(fh)
    n_d_by_row = {}
    for fname, blob in data.items():
        if blob['f'] != f_target:
            continue
        for r in blob['matrices']:
            d = r.get('decompose', {})
            if d.get('success'):
                n_d_by_row[r['row']] = d['N_D']
    pieces = []
    for (ridx, theta, groups) in rows:
        if ridx not in n_d_by_row:
            continue
        V = ig.build_ring(groups, f_hdr)
        pieces.append({
            'row': ridx, 'theta': theta,
            'V_01': V, 'N_D': n_d_by_row[ridx],
        })
    return pieces


def run_trial(pieces, K, rng, verbose=False):
    """Pick K pieces, alternate (0,1)/(1,2) subspaces, compose, reduce."""
    picks = rng.sample(range(len(pieces)), K)
    pieces_picked = [pieces[p] for p in picks]
    subspaces = [(0,1) if i % 2 == 0 else (1,2) for i in range(K)]

    naive_nd = sum(p['N_D'] for p in pieces_picked)
    pretty_picks = [(p['row'], p['theta'], sub, p['N_D'])
                    for p, sub in zip(pieces_picked, subspaces)]

    ring_mats = []
    for p, sub in zip(pieces_picked, subspaces):
        V = p['V_01']
        if sub == (1, 2):
            V = conjugate_by_X(V)
        ring_mats.append(V)

    V = ring_mats[0]
    for M in ring_mats[1:]:
        V = cr._fast_matmul(V, M)

    t0 = time.time()
    result = cr.decompose_canonical(V, verbose=verbose, greedy_single=True)
    decomp_wall = time.time() - t0
    if not result['success']:
        return {
            'ok': False, 'naive_nd': naive_nd, 'reduced_nd': None,
            'wall': decomp_wall, 'picks': pretty_picks,
            'error': result.get('error', 'unknown decomp failure'),
            'sde_initial': result.get('sde_chi_initial'),
            'sde_final': result.get('sde_chi_final'),
        }
    reduced_nd = result['D_count']
    return {
        'ok': True, 'naive_nd': naive_nd, 'reduced_nd': reduced_nd,
        'ratio': reduced_nd / naive_nd, 'wall': decomp_wall,
        'picks': pretty_picks,
        'sde_initial': result['sde_chi_initial'],
        'sde_final': result['sde_chi_final'],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--f', type=int, default=10,
                    help='Nick fits level (4,6,8,10,12,14,16)')
    ap.add_argument('--K', type=int, default=5,
                    help='Number of R_z pieces per trial')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    fits_path = _HERE / f"fits_f={args.f}.txt"
    # Find the JSON containing this f
    json_candidates = [
        _HERE / "nick_decompositions.json",
        _HERE / "nick_decompositions_f468.json",
        _HERE / "nick_decompositions_f16_n150.json",
        _HERE / "nick_decompositions_f16.json",
    ]
    json_path = None
    for jp in json_candidates:
        if not jp.exists():
            continue
        with open(jp) as fh:
            data = json.load(fh)
        if any(blob['f'] == args.f for blob in data.values()):
            json_path = jp
            break
    if json_path is None:
        sys.exit(f"no decomposition JSON found for f={args.f}")

    print(f"[setup] fits = {fits_path.name}, decomp JSON = {json_path.name}, "
          f"K = {args.K}, trials = {args.trials}, seed = {args.seed}")
    cr.get_prefix_table()  # warm
    pieces = load_nick_pieces(fits_path, json_path, args.f)
    print(f"[setup] loaded {len(pieces)} Nick pieces at f={args.f}, "
          f"N_D mean={sum(p['N_D'] for p in pieces)/len(pieces):.1f}")

    rng = random.Random(args.seed)
    results = []
    for ti in range(args.trials):
        r = run_trial(pieces, args.K, rng, verbose=args.verbose)
        if r['ok']:
            print(f"trial {ti:2d}: naive={r['naive_nd']:4d}  "
                  f"reduced={r['reduced_nd']:4d}  "
                  f"ratio={r['ratio']:.3f}  "
                  f"sde {r['sde_initial']}->{r['sde_final']}  "
                  f"wall={r['wall']:.1f}s")
        else:
            print(f"trial {ti:2d}: FAILED  naive={r['naive_nd']}  "
                  f"sde {r['sde_initial']}->{r['sde_final']}  "
                  f"err={r['error'][:60]}  wall={r['wall']:.1f}s")
        results.append(r)

    ok = [r for r in results if r['ok']]
    if ok:
        ratios = [r['ratio'] for r in ok]
        avg = sum(ratios) / len(ratios)
        savings = [(r['naive_nd'] - r['reduced_nd']) for r in ok]
        print(f"\n=== summary ===")
        print(f"successful trials: {len(ok)}/{len(results)}")
        print(f"ratio mean:        {avg:.3f}  (min {min(ratios):.3f}, "
              f"max {max(ratios):.3f})")
        print(f"absolute savings:  mean {sum(savings)/len(savings):.1f}  "
              f"D-gates per trial")
        if avg < 0.7:
            print("VERDICT: SMOKING GUN — substantial recompile savings")
        elif avg < 0.95:
            print("VERDICT: real but modest savings")
        else:
            print("VERDICT: no meaningful savings; tracks naive Euler")
    else:
        print("\n=== summary ===\nNO successful trials")


if __name__ == '__main__':
    main()
