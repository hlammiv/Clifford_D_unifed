"""Phase-sensitive Clifford invariant ('triple invariant') for isometries J (D x 3):
chi = Choi state on ref (x) sys (x) ancillas; c(g) = <chi|g|chi> for all canonical Paulis g = X^a Z^b.
Phi(g1,g2) = c(g1) c(g2) conj(c(g3)) w^{-beta}, where g1 g2 = w^beta g3.  Under any Clifford on
(sys+ancillas) and any Clifford on ref (i.e. right logical Clifford G), plus global phase, the multiset
{Phi(g1,g2)} is invariant (Clifford phase conventions cancel).  Strictly stronger than |c|^2."""
import numpy as np, itertools, hashlib
z = np.exp(2j*np.pi/9); w = z**3
class Paulis:
    def __init__(self, nq):
        self.nq = nq; self.D = 3**nq
        self.labs = np.array(list(itertools.product(range(3), repeat=2*nq)))   # (a1,b1,...)
        self.a = self.labs[:, 0::2]; self.b = self.labs[:, 1::2]
        self.digits = np.array(list(itertools.product(range(3), repeat=nq)))
        pw = 3**np.arange(nq)[::-1]
        self.dst = ((self.digits[None, :, :] + self.a[:, None, :]) % 3) @ pw          # (Ng, D)
        self.ph = w**((self.digits[None, :, :] * self.b[:, None, :]).sum(-1) % 3)    # (Ng, D)
        # product table
        lp = 3**np.arange(2*nq)[::-1]
        self.lin = self.labs @ lp
    def cvec(self, chi):
        """chi (N,D) -> c (N,Ng) = <chi| g |chi>, g|r> = ph[r] |r+a>."""
        # <chi|g|chi> = sum_r conj(chi[dst[r]]) ph[r] chi[r]
        return np.einsum('ngr,gr,nr->ng', np.conj(chi[:, self.dst]), self.ph, chi, optimize=True)
    def prod(self, i, j):
        """indices arrays -> (k, beta) with g_i g_j = w^beta g_k"""
        a1, b1 = self.a[i], self.b[i]; a2, b2 = self.a[j], self.b[j]
        beta = (b1*a2).sum(-1) % 3
        a3 = (a1+a2) % 3; b3 = (b1+b2) % 3
        lab = np.empty((len(i), 2*self.nq), int); lab[:, 0::2] = a3; lab[:, 1::2] = b3
        k = lab @ (3**np.arange(2*self.nq)[::-1])
        return k, beta
def choi(J):
    """J (D,3) -> chi on ref (x) J-space, ref is the FIRST qutrit."""
    return (np.eye(3)[:, :, None] * J.T[None, :, :]).transpose(0, 1, 2).reshape(-1)[None] if False else \
        np.concatenate([J[:, i] for i in range(3)])[None]/np.sqrt(3)
def triple_hash(J, PS, q=1e6):
    chi = choi(J)
    c = PS.cvec(chi)[0]
    S = np.nonzero(np.abs(c) > 1e-9)[0]
    I, Jj = np.meshgrid(S, S, indexing='ij'); I = I.ravel(); Jj = Jj.ravel()
    k, beta = PS.prod(I, Jj)
    val = c[I]*c[Jj]*np.conj(c[k])*w**(-beta)
    keep = np.abs(val) > 1e-12
    v = np.stack([np.rint(val[keep].real*q), np.rint(val[keep].imag*q)], 1).astype(np.int64)
    v = v[np.lexsort(v.T[::-1])]
    marg = float(np.min(0.5-np.abs(np.concatenate([val[keep].real, val[keep].imag])*q - np.concatenate([val[keep].real*0+np.rint(val[keep].real*q), np.rint(val[keep].imag*q)]))) ) if keep.any() else 0.5
    return hashlib.blake2b(v.tobytes(), digest_size=8).hexdigest(), len(S), marg
