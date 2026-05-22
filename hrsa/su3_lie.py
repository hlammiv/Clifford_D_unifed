"""SU(3) Lie algebra utilities: Gell-Mann basis, log/exp via principal branch."""
import numpy as np
from scipy.linalg import logm, expm


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
    """Return real coefs c_a with log(U) = i * sum_a c_a * lambda_a."""
    # NOTE: scipy.linalg.logm uses the principal branch and can be sign-ambiguous
    # for U with eigenvalues near -1; safe for U close to identity, which is our regime.
    L = logm(U)
    coefs = np.empty(8, dtype=float)
    for a in range(8):
        coefs[a] = (np.trace(GELL_MANN[a] @ L) / (2.0j)).real
    return coefs


def matrix_exp_su3(coefs):
    """Return exp(i * sum_a c_a * lambda_a) for length-8 real coefs."""
    H = np.zeros((3, 3), dtype=complex)
    for a in range(8):
        H = H + coefs[a] * GELL_MANN[a]
    return expm(1j * H)


def exp_log_roundtrip_check(U):
    """Frobenius residual |exp(log(U)) - U|_F via the Gell-Mann coordinate detour."""
    return float(np.linalg.norm(matrix_exp_su3(matrix_log_su3(U)) - U, ord='fro'))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1: identity -> zero coefs
    I = np.eye(3, dtype=complex)
    c_I = matrix_log_su3(I)
    print(f"Test 1 (identity): |coefs| = {np.linalg.norm(c_I):.3e}")

    # Test 2: deliberate 2-parameter U
    target = np.array([0.3, 0, 0.5, 0, 0, 0, 0, 0])
    U2 = matrix_exp_su3(target)
    c2 = matrix_log_su3(U2)
    print(f"Test 2 (2-param): coefs recovered = {c2}")
    print(f"Test 2 (2-param): |coefs - target| = {np.linalg.norm(c2 - target):.3e}")
    print(f"Test 2 (2-param): roundtrip residual = {exp_log_roundtrip_check(U2):.3e}")

    # Test 3: 5 random near-identity unitaries
    print("Test 3 (random near-identity):")
    for k in range(5):
        c_rand = 0.1 * rng.standard_normal(8)
        U_rand = matrix_exp_su3(c_rand)
        res = exp_log_roundtrip_check(U_rand)
        c_rec = matrix_log_su3(U_rand)
        coef_err = np.linalg.norm(c_rec - c_rand)
        print(f"  k={k}: roundtrip residual = {res:.3e}, |coef_err| = {coef_err:.3e}")
