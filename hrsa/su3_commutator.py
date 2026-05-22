"""SU(3) group-commutator factorization for Solovay-Kitaev recursion.

Given E close to I in SU(3) with |E - I|_F = eps, find A, B in SU(3) with
[A, B] = A B A^dagger B^dagger approximately equal to E and
|A - I|_F, |B - I|_F = O(sqrt(eps)).

Strategy: analytic Lie-algebra seed (diagonalize log(E)/i, split the diagonal
into two SU(2) sub-block commutators), then polish with scipy.optimize over the
8+8 Gell-Mann coordinates of X, Y where A=exp(iX), B=exp(iY).
"""
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

try:
    from su3_lie import GELL_MANN, matrix_log_su3, matrix_exp_su3
except ImportError:
    # Minimal fallback if su3_lie.py is not on the path.
    def _gell_mann():
        l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
        l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
        l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
        l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
        l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
        l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
        l8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3.0)
        return (l1, l2, l3, l4, l5, l6, l7, l8)
    GELL_MANN = _gell_mann()

    def matrix_log_su3(U):
        L = logm(U)
        coefs = np.empty(8, dtype=float)
        for a in range(8):
            coefs[a] = (np.trace(GELL_MANN[a] @ L) / (2.0j)).real
        return coefs

    def matrix_exp_su3(coefs):
        H = np.zeros((3, 3), dtype=complex)
        for a in range(8):
            H = H + coefs[a] * GELL_MANN[a]
        return expm(1j * H)


GM = GELL_MANN  # shorthand


def group_commutator(A, B):
    """Group commutator A B A^dagger B^dagger."""
    return A @ B @ A.conj().T @ B.conj().T


def _coefs_to_H(c):
    """Build Hermitian traceless H = sum_a c_a lambda_a from 8 real coefs."""
    H = np.zeros((3, 3), dtype=complex)
    for a in range(8):
        H = H + c[a] * GM[a]
    return H


def _H_to_coefs(H):
    """Project Hermitian traceless H onto Gell-Mann basis: c_a = (1/2) Tr(lambda_a H)."""
    c = np.empty(8, dtype=float)
    for a in range(8):
        c[a] = 0.5 * np.trace(GM[a] @ H).real
    return c


def _analytic_seed(E):
    """Construct X, Y in Gell-Mann coordinates from a diagonalized-D ansatz.

    Diagonalize H = log(E)/i = U D U^dagger with D = diag(d0, d1, d2).
    In the eigenbasis, want [X', Y'] = -i D.
    Split D = alpha * diag(1,-1,0) + beta * diag(0,1,-1) with alpha = d0, beta = -d2.
    Use X' = x1 lambda_1 + x6 lambda_6, Y' = y2 lambda_2 + y7 lambda_7.
    Leading-order solution (ignoring cross-commutators in lambda_4 direction):
       2 x1 y2 = -alpha,  2 x6 y7 = -beta.
    Pick balanced |x| = |y| ~ sqrt(|coef|/2).
    Rotate back: X = U X' U^dagger, Y = U Y' U^dagger.
    """
    # log(E) / i, projected to Hermitian traceless (anti-Hermitian piece of logm/i).
    L = logm(E)
    H = (L - L.conj().T) / (2.0j)  # Hermitian
    # Remove residual trace (E in SU(3) should have traceless log already).
    H = H - (np.trace(H).real / 3.0) * np.eye(3)

    # Diagonalize.
    w, U = np.linalg.eigh(H)  # H = U diag(w) U^dagger
    d0, d1, d2 = w[0], w[1], w[2]
    alpha = d0
    beta = -d2

    # x1 y2 = -alpha/2.  Pick x1 = sign_a * sqrt(|alpha|/2), y2 = -sign_a * sqrt(|alpha|/2)*sign(alpha)?
    # Simpler: x1 = sqrt(|alpha|/2), y2 = -sign(alpha) * sqrt(|alpha|/2).
    def _pair(c):
        s = np.sign(c) if c != 0 else 1.0
        m = np.sqrt(abs(c) / 2.0)
        return m, -s * m

    x1, y2 = _pair(alpha)
    x6, y7 = _pair(beta)

    Xp = x1 * GM[0] + x6 * GM[5]  # lambda_1 is GM[0], lambda_6 is GM[5]
    Yp = y2 * GM[1] + y7 * GM[6]  # lambda_2 is GM[1], lambda_7 is GM[6]

    X = U @ Xp @ U.conj().T
    Y = U @ Yp @ U.conj().T

    cX = _H_to_coefs(X)
    cY = _H_to_coefs(Y)
    return cX, cY


def _commutator_residual(params, E):
    cX = params[:8]
    cY = params[8:]
    A = matrix_exp_su3(cX)
    B = matrix_exp_su3(cY)
    G = group_commutator(A, B)
    R = G - E
    # Frobenius squared.
    return float(np.vdot(R.flatten(), R.flatten()).real)


def factor_commutator(E, polish=True, max_iter=400, verbose=False):
    """Return (A, B, residual) with A B A^dagger B^dagger approx E in SU(3).

    Pipeline:
      1) Analytic seed from log(E)/i diagonalization.
      2) Optional scipy.optimize.minimize (L-BFGS-B) polish on 16 real coords.

    residual = |group_commutator(A, B) - E|_F.
    """
    cX0, cY0 = _analytic_seed(E)
    params = np.concatenate([cX0, cY0])

    if verbose:
        r0 = np.sqrt(_commutator_residual(params, E))
        print(f"  seed residual = {r0:.3e}")

    if polish:
        res = minimize(
            _commutator_residual,
            params,
            args=(E,),
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-24, "gtol": 1e-14},
        )
        params = res.x
        if verbose:
            r1 = np.sqrt(_commutator_residual(params, E))
            print(f"  polished residual = {r1:.3e}, nit={res.nit}")

    cX = params[:8]
    cY = params[8:]
    A = matrix_exp_su3(cX)
    B = matrix_exp_su3(cY)
    residual = float(np.linalg.norm(group_commutator(A, B) - E, ord="fro"))
    return A, B, residual


def _random_near_identity(eps, rng):
    """Random E = exp(i * eps * H) with H Hermitian traceless unit-normalized."""
    c = rng.standard_normal(8)
    c = c / np.linalg.norm(c)
    return matrix_exp_su3(eps * c)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Sanity: group_commutator(I, I) = I.
    I = np.eye(3, dtype=complex)
    assert np.allclose(group_commutator(I, I), I)

    # Sanity: group_commutator is unitary when A, B are.
    A = matrix_exp_su3(0.1 * rng.standard_normal(8))
    B = matrix_exp_su3(0.1 * rng.standard_normal(8))
    G = group_commutator(A, B)
    assert np.linalg.norm(G @ G.conj().T - I, ord="fro") < 1e-10

    print(f"{'eps':>8s} {'|E-I|_F':>10s} {'|A-I|_F':>10s} {'|B-I|_F':>10s} "
          f"{'residual':>12s} {'ratio':>10s} {'pass':>6s}")
    print("-" * 78)

    overall_pass = True
    for eps in [0.5, 0.1, 0.01]:
        for trial in range(3):
            E = _random_near_identity(eps, rng)
            e_dist = float(np.linalg.norm(E - I, ord="fro"))
            A, B, residual = factor_commutator(E)
            a_dist = float(np.linalg.norm(A - I, ord="fro"))
            b_dist = float(np.linalg.norm(B - I, ord="fro"))
            ratio = residual / e_dist if e_dist > 0 else float("inf")
            # Pass criterion for eps in {0.1, 0.01}: ratio < 0.5.
            ok = (eps == 0.5) or (ratio < 0.5)
            if not ok:
                overall_pass = False
            print(f"{eps:>8.3f} {e_dist:>10.3e} {a_dist:>10.3e} {b_dist:>10.3e} "
                  f"{residual:>12.3e} {ratio:>10.3e} {'OK' if ok else 'FAIL':>6s}")

    print("-" * 78)
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
