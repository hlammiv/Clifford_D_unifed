"""ct_frontier.py — exact optimal T-count frontier for R_z(θ) in ancilla-free single-qutrit Clifford+T.

Every single-qutrit C+T operator of T-count <= t is C·T(P_t)···T(P_1) with C one of the 216 projective
Cliffords and T(P) one of 8 Clifford-conjugated T rotations (4 axes x {T,T^dag}), consecutive axes
distinct (Glaudell–Ross–Taylor 1803.05047: (216/5)(8·6^t - 3) distinct operators).
For every operator we compute the phase-minimised Frobenius distance to the curve R_z(θ)=diag(e^{-iθ/2},e^{iθ/2},1):
    d(U,θ)^2 = 6 - 2|u00 e^{iθ/2} + u11 e^{-iθ/2} + u22|
and record, on a θ grid, ε*(θ,t) = min over T-count<=t.  Also counts operators within δ of the curve
vs. the Haar prediction N·vol(tube_δ)/vol(PU(3)) (equidistribution check).
Usage: python3 ct_frontier.py --tmax 7"""
import argparse, json, math, time
import numpy as np
z = np.exp(2j * np.pi / 9); w = z ** 3
H = np.array([[w ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(-3 + 0j)
S = np.diag([1, 1, w]).astype(complex)
T = np.diag([1, z, z ** 8])
X = np.roll(np.eye(3), 1, axis=0).astype(complex); Z = np.diag([1, w, w * w])

def key(U):
    i = np.argmax(np.abs(U.ravel()) > 1e-6)
    V = U * (abs(U.ravel()[i]) / U.ravel()[i])
    v = V.ravel()
    return np.round(np.concatenate([v.real, v.imag]) * 1e4 + 0.123).astype(np.int64).tobytes()

def clifford_group():
    seen = {key(np.eye(3)): np.eye(3, dtype=complex)}; frontier = [np.eye(3, dtype=complex)]
    while frontier:
        nf = []
        for U in frontier:
            for G in (H, S):
                V = G @ U; k = key(V)
                if k not in seen:
                    seen[k] = V; nf.append(V)
        frontier = nf
    return np.array(list(seen.values()))

def rotations(C):
    axes = [Z, X, X @ Z, X @ Z @ Z]
    rots, ax = [], []
    for a, P in enumerate(axes):
        for c in C:
            Q = c @ Z @ c.conj().T
            if abs(abs(np.vdot(P, Q)) - 3) < 1e-8:
                break
        R = c @ T @ c.conj().T
        rots += [R, R.conj().T]; ax += [a, a]
    return np.array(rots), np.array(ax)

def curve_dist(U, thetas):
    """U: (n,3,3); returns (n, len(thetas)) phase-min Frobenius distances."""
    e1 = np.exp(0.5j * thetas); e2 = np.conj(e1)
    tr = U[:, 0, 0, None] * e1 + U[:, 1, 1, None] * e2 + U[:, 2, 2, None]
    return np.sqrt(np.maximum(6 - 2 * np.abs(tr), 0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tmax", type=int, default=7)
    ap.add_argument("--ntheta", type=int, default=4096); ap.add_argument("--out", default="ct_frontier.json")
    a = ap.parse_args()
    C = clifford_group(); assert len(C) == 216, len(C)
    R, RA = rotations(C)
    thetas = np.linspace(0, 4 * np.pi, a.ntheta, endpoint=False)
    best = curve_dist(C, thetas).min(axis=0)          # t = 0
    volPU3 = 16 * math.sqrt(3) * math.pi ** 5 / 3
    V7 = 16 * math.pi ** 3 / 105; L = 4 * math.pi / math.sqrt(2)
    res = [dict(t=0, N=216, eps_median=float(np.median(best)), eps_max=float(best.max()))]
    print(res[-1], flush=True)
    W = np.eye(3, dtype=complex)[None]; WA = np.array([-1])
    Ntot = 216
    for t in range(1, a.tmax + 1):
        t0 = time.time()
        nW, nA = [], []
        for r, ra in zip(R, RA):
            m = WA != ra
            nW.append(np.einsum("ij,njk->nik", r, W[m])); nA.append(np.full(m.sum(), ra))
        W = np.concatenate(nW); WA = np.concatenate(nA)
        Ntot += 216 * len(W)
        # distance: off-diagonal prefilter, then full θ grid for survivors
        dmax = max(float(np.max(best)) * 1.0, 1e-9)
        near = {d: 0 for d in (0.2, 0.3, 0.4)}
        chunk = max(1, 400000 // 216)
        for s in range(0, len(W), chunk):
            U = np.einsum("cij,njk->cnik", C, W[s:s + chunk]).reshape(-1, 3, 3)
            off = np.sum(np.abs(U) ** 2, axis=(1, 2)) - np.sum(np.abs(np.diagonal(U, axis1=1, axis2=2)) ** 2, axis=1)
            keep = off < dmax ** 2
            if keep.any():
                D = curve_dist(U[keep], thetas)
                best = np.minimum(best, D.min(axis=0))
                dmin = D.min(axis=1)
                for d in near:
                    near[d] += int((dmin < d).sum())
        pred = {d: Ntot * L * V7 * d ** 7 / volPU3 for d in near}
        row = dict(t=t, N=Ntot, N_formula=216 * (8 * 6 ** t - 3) / 5, eps_median=float(np.median(best)),
                   eps_max=float(best.max()), eps_p10=float(np.quantile(best, 0.1)),
                   tube_counts={str(d): near[d] for d in near}, tube_pred_haar={str(d): pred[d] for d in near},
                   sec=time.time() - t0)
        res.append(row); print(json.dumps(row), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
