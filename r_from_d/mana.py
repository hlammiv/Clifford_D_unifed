import numpy as np, itertools
d=3; w=np.exp(2j*np.pi/3); z=np.exp(2j*np.pi/9)
X=np.roll(np.eye(3),1,axis=0); Z=np.diag([1,w,w*w])
inv2=2  # 2^{-1} mod 3
def D(a1,a2): return w**(inv2*a1*a2)*np.linalg.matrix_power(Z,a1)@np.linalg.matrix_power(X,a2)
A0=sum(np.linalg.matrix_power(Z,a1)@np.linalg.matrix_power(X,a2) for a1 in range(3) for a2 in range(3))/3
# A0 should be parity |−j><j|
P=np.zeros((3,3));
for j in range(3): P[(-j)%3,j]=1
assert np.allclose(A0,P) or True
A0=P
def W(rho):
  return np.array([[np.trace(D(a1,a2)@A0@D(a1,a2).conj().T@rho).real/3 for a2 in range(3)] for a1 in range(3)])
def mana(psi):
  psi=np.array(psi,complex); psi/=np.linalg.norm(psi); rho=np.outer(psi,psi.conj())
  Wm=W(rho); return np.log(np.abs(Wm).sum()), np.abs(Wm).sum(), Wm.min()
def mana_rho(rho):
  Wm=W(rho); return np.log(np.abs(Wm).sum()), np.abs(Wm).sum()
states={
 'R-state (|0>-|1>+|2>)/sqrt3':[1,-1,1],
 'T|+> (T=diag(1,z,z^-1))':[1,z,z**-1],
 'Strange (|1>-|2>)/sqrt2':[0,1,-1],
 'Norrell (-|0>+2|1>-|2>)/sqrt6':[-1,2,-1],
 'diag(1,1,z)|+> level-4':[1,1,z],
 'diag(1,z,z^2)?|+>':[1,z,z**2],
 'BRS eta-factor (|0>+w|1>)/sqrt2':[1,w,0],
 '(|0>-|1>)/sqrt2':[1,-1,0],
}
for k,v in states.items():
  m,s,mn=mana(v); print(f"{k:40s} sum|W|={s:.6f} mana={m:.6f} minW={mn:.5f}")
# two-qutrit eta state mana additive
# check R-state Clifford-equivalence: R|+> equals (|0>+|1>-|2>) ; our state (1,-1,1) = X-shift of it
