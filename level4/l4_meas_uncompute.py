"""l4_meas_uncompute.py — level-4 diag(1,1,ζ) with 4 T using measurement-based uncomputation.

Circuit (qutrit 0 = system, qutrit 1 = clean ancilla |0>):
  C2X (3 T, GRVY |2>-controlled X)      : |x>|0> -> |x>|b>,  b = [x == 2]
  T on ancilla (1 T)                    : phase ζ^{t(b)}, t(0)=0, t(1)=1
  measure ancilla in the Fourier basis  : outcome m in {0,1,2}, prob 1/3 each
  Clifford correction on the system     : diag(1,1,ω^{m'}) (the measurement leaves phase ω^{-m b})
Checks that, for every outcome m, the post-measurement system state equals
diag(1,1,ζ)|ψ> exactly after a SINGLE-QUTRIT DIAGONAL CLIFFORD correction, so the
gate is deterministic with T-count 4 (vs 7 unitary)."""
import numpy as np
exec(open("l4_from_8T.py").read().split("# 2)")[0])   # builds c2x (3-T |2>-controlled X), z, w, I3, ...
F = np.array([[w ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3)   # Fourier basis
target = np.diag([1, 1, z])
rng = np.random.default_rng(5)
worst = 0.0
for trial in range(200):
    psi = rng.normal(size=3) + 1j * rng.normal(size=3); psi /= np.linalg.norm(psi)
    state = (c2x.conj().T if False else c2x) @ np.kron(psi, [1, 0, 0])      # compute
    state = np.kron(I3, np.diag([1, z, z ** 8])) @ state                     # T on ancilla
    for m in range(3):
        bra = F[:, m].conj()                                                 # <f_m| on ancilla
        sys = np.array([sum(bra[a] * state[3 * x + a] for a in range(3)) for x in range(3)])
        p = np.vdot(sys, sys).real
        assert abs(p - 1 / 3) < 1e-9, p
        sys /= np.sqrt(p)
        # find the diagonal Clifford correction diag(1,1,ω^c) (up to global phase) that gives target ψ
        best = min((np.linalg.norm(np.diag([1, 1, w ** c]) @ sys
                                   - np.vdot(target @ psi, np.diag([1, 1, w ** c]) @ sys) * (target @ psi)), c)
                   for c in range(3))
        worst = max(worst, best[0])
        if trial == 0:
            print(f"outcome m={m}: prob {p:.4f}, correction diag(1,1,ω^{best[1]}), residual {best[0]:.1e}")
print(f"all 200 states x 3 outcomes: max residual {worst:.1e}  ->",
      "DETERMINISTIC level-4 with 4 T + measurement + Clifford feedforward" if worst < 1e-9 else "FAILS")
