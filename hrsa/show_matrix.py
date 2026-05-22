#!/usr/bin/env python3
"""
show_matrix.py  --  Parse an HRSA output file and display the target unitary
R_{(0,1)}^Z(theta), the approximate unitary X_{(0,1)} H, and their difference.

Usage:
    python3 show_matrix.py <output_file> [output_file2 ...]

The Householder reflector H = I - u * u^dagger is built from the three complex
entries u = [x1, x2, x3] printed in the output file.  The correction matrix
X_{(0,1)} = [[0,1,0],[1,0,0],[0,0,1]] (row swap) is then applied to give the
approximation X_{(0,1)} H ~ R_{(0,1)}^Z(theta) = Diag(e^{-i*theta/2}, e^{i*theta/2}, 1).
"""

import sys
import re
import math
import cmath
import numpy as np


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_output_file(path):
    """
    Parse an HRSA output file and return a dict with:
        theta   : float   -- rotation angle (external, before HRSA negation)
        u       : (3,) ndarray of complex -- the Householder vector [x1, x2, x3]
        epsilon : float   -- target epsilon
        f       : int     -- f-level at which the solution was found (or None)
        success : bool
        frob_vec: float   -- vector-level eps_diff reported by epsTest
        frob_mat: float   -- matrix Frobenius distance reported by matrixFrobeniusCheck
        vec_norm_sq: float -- |u|^2 reported in the file
    """
    with open(path) as fh:
        text = fh.read()

    result = {}

    # ── theta and epsilon: prefer filename encoding ────────────────────────────
    # Filename form: out_a<THETA>_e<EPS>_f<MAXF>
    fname = path.split('/')[-1]
    m = re.search(r'out_a([-\d.]+)_e([\d.]+)_f(\d+)', fname)
    if m:
        result['theta']   = float(m.group(1))
        result['epsilon'] = float(m.group(2))
        result['max_f']   = int(m.group(3))
    else:
        result['theta']   = None
        result['epsilon'] = None
        result['max_f']   = None

    # 'Target x_1 = e^(i theta/2) : RE + iIM'
    # gives e^{i*theta/2}, so theta = 2 * atan2(im, re)
    m = re.search(r'Target x_1.*?: ([-\d.e+]+) \+ i([-\d.e+]+)', text)
    if m:
        target_x1 = complex(float(m.group(1)), float(m.group(2)))
        if result['theta'] is None:
            result['theta'] = 2.0 * math.atan2(target_x1.imag, target_x1.real)

    # Fall back: recover epsilon from 'Eps. Cond.' line (eps_cond = eps^2 / 8)
    if result['epsilon'] is None:
        m = re.search(r'Eps\. Cond\.: ([\d.e+\-]+)', text)
        if m:
            result['epsilon'] = math.sqrt(8.0 * float(m.group(1)))

    if result['theta'] is None:
        raise ValueError(f"Could not determine theta from {path}")

    # ── u = [x1, x2, x3] from the 'Found' lines ──────────────────────────────
    u = []
    for label in ['Found x_1', 'Found x_2', 'Found x_3']:
        m = re.search(label + r'.*?: ([-\d.e+]+) \+ i([-\d.e+]+)', text)
        if not m:
            raise ValueError(f"Could not find '{label}' in {path}")
        u.append(complex(float(m.group(1)), float(m.group(2))))
    result['u'] = np.array(u, dtype=complex)

    # ── f level: last 'for f = N' line ────────────────────────────────────────
    fs = re.findall(r'for f = (\d+)', text)
    result['f'] = int(fs[-1]) if fs else None

    # ── success ───────────────────────────────────────────────────────────────
    result['success'] = 'Success!' in text

    # ── vector eps_diff ───────────────────────────────────────────────────────
    m = re.search(r'Epsilon Diff\. Val\.: ([\d.e+\-]+)', text)
    result['frob_vec'] = float(m.group(1)) if m else None

    # ── matrix Frobenius distance (from matrixFrobeniusCheck) ─────────────────
    m = re.search(r'Matrix Frobenius distance.*?= ([\d.e+\-]+)', text)
    result['frob_mat'] = float(m.group(1)) if m else None

    # ── vector norm (should be 2) ─────────────────────────────────────────────
    m = re.search(r'Frobenius norm of vector.*?: ([\d.e+\-]+)', text)
    result['vec_norm_sq'] = float(m.group(1)) if m else None

    return result


# ── Matrix construction ───────────────────────────────────────────────────────

def build_matrices(theta, u):
    """
    Given theta (external rotation angle) and u = [x1, x2, x3] (complex vector),
    return (target, approx, error) as 3x3 complex ndarrays.

    target = R_{(0,1)}^Z(theta)  = Diag(e^{-i*theta/2}, e^{i*theta/2}, 1)
    H      = I_3 - u * u^dagger  (Householder reflector)
    approx = X_{(0,1)} * H       where X_{(0,1)} = [[0,1,0],[1,0,0],[0,0,1]]
    error  = approx - target
    """
    target = np.diag([cmath.exp(-1j * theta / 2),
                      cmath.exp( 1j * theta / 2),
                      1.0 + 0j])

    H = np.eye(3, dtype=complex) - np.outer(u, np.conj(u))

    # X_{(0,1)} swaps rows 0 and 1; applying it is just reordering H's rows
    approx = H[[1, 0, 2], :]

    error = approx - target
    return target, approx, error


# ── Pretty printing ───────────────────────────────────────────────────────────

def _fmt(z, width=26):
    """Format a complex number, suppressing negligibly small parts."""
    re, im = z.real, z.imag
    tol = 5e-15 * max(abs(re), abs(im), 1.0)
    if abs(im) < tol:
        s = f'{re:+.8f}        '
    elif abs(re) < tol:
        s = f'  {im:+.8f}j      '
    else:
        sign = '+' if im >= 0 else '-'
        s = f'{re:+.8f} {sign} {abs(im):.8f}j'
    return s.center(width)


def print_matrix(M, title, indent='  '):
    print(f'{indent}{title}')
    for row in M:
        print(f'{indent}  [ ' + '  '.join(_fmt(z) for z in row) + ' ]')
    print()


def print_report(path, info):
    theta   = info['theta']
    u       = info['u']
    epsilon = info['epsilon']
    f       = info['f']

    target, approx, error = build_matrices(theta, u)
    frob = np.linalg.norm(error, 'fro')
    passes = (frob < epsilon) if epsilon is not None else None

    print('=' * 80)
    print(f'  File    : {path}')
    print(f'  theta   : {theta:.10f}  rad  ({math.degrees(theta):.6f} deg)')
    print(f'  epsilon : {epsilon}')
    print(f'  f level : {f}')
    print(f'  Success : {info["success"]}')
    print(f'  |u|^2   : {info["vec_norm_sq"]}  (should be exactly 2)')
    print()

    print_matrix(target, 'Target   R_(0,1)^Z(theta)  =  Diag( e^{-i*theta/2},  e^{+i*theta/2},  1 )')
    print_matrix(approx, 'Approx   X_(0,1) * (I - u u†)')
    print_matrix(error,  'Error    (Approx - Target)')

    print(f'  Frobenius distance  ||Approx - Target||_F  =  {frob:.8e}')
    if epsilon is not None:
        print(f'  Target epsilon                               {epsilon}')
        print(f'  Passes (frob < epsilon)                      {"YES" if passes else "NO"}')
    if info['frob_mat'] is not None:
        delta = abs(frob - info['frob_mat'])
        print(f'  File-reported Frobenius: {info["frob_mat"]:.8e}  (diff from recomputed: {delta:.1e})')
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            info = parse_output_file(path)
            print_report(path, info)
        except Exception as exc:
            print(f'ERROR processing {path}: {exc}', file=sys.stderr)
