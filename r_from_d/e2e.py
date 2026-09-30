import numpy as np, itertools
z=np.exp(2j*np.pi/9); w=z**3; I3=np.eye(3)
Hd=np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j)
X=np.roll(np.eye(3),1,axis=0)
def CX(k):  # |x,y> -> |x, y + k x>
  M=np.zeros((9,9))
  for x in range(3):
    for y in range(3): M[3*x+(y+k*x)%3,3*x+y]=1
  return M
def Dg(g): return np.diag([z**v for v in g])
# gadget realization: g on line (a,b): use CX(k) to map y-> k x + y ... line (0,1): D on y; (1,1): CX(1); (1,2): value x+2y = 2(2x+y): CX(2), D with permuted table
def gadget(line,g):
  a,b=line
  if line==(0,1): return np.kron(I3,Dg(g))
  if line==(1,1): return CX(1).T@np.kron(I3,Dg(g))@CX(1)
  if line==(1,2): # x+2y = 2*(2x+y)
    gg=[g[(2*t)%3] for t in range(3)]
    return CX(2).T@np.kron(I3,Dg(gg))@CX(2)
choice={(0,1):(0,1,1),(1,1):(0,0,2),(1,2):(0,0,2)}
Gd=np.eye(9,dtype=complex)
for l,g in choice.items(): Gd=gadget(l,g)@Gd
target=np.diag([z**((7*(x==2)+3*((x==2) and (y!=0)))%9) for x in range(3) for y in range(3)])
ratio=np.diag(target)/np.diag(Gd); ex=np.round(np.angle(ratio)/(2*np.pi/9))%9
print('residual exponents (x,y order):',ex.reshape(3,3))
# residual must be Clifford diag: w^{quadratic}; fix it exactly using it as Clifford phase (it is multiple of 3)
Cz=Gd@np.diag(ratio)   # = target exactly (ratio is Clifford diag)
Z11=np.diag([1,w,w]); CZc=Cz   # C2(zeta^7 Z(1,1)) with D-count 3
CXc=np.kron(I3,Hd)@CZc@np.kron(I3,Hd.conj().T)
U1=np.kron(I3,Z11)
best=None
for seq in itertools.permutations(['X','Z','X']):
  pass
# GRVY order (left-to-right time): Z11, C[X], C[Z], Z11, C[X], Z11 ; then S^dag on control
M=U1@CXc@U1@CZc@CXc@U1
M=np.kron(np.diag([1,1,w**2]),I3)@M
t12=np.array([[1,0,0],[0,0,1],[0,1,0]])
C2mt=np.kron(np.diag([1,1,0]),I3)+np.kron(np.diag([0,0,1]),-t12)
print('equals C2(-tau12)?', np.allclose(M,C2mt), 'up to phase?', np.allclose(M/M[0,0],C2mt))
if not np.allclose(M,C2mt):
  for s in [np.diag([1,1,w**k]) for k in range(3)]:
    for Xv in (CXc, np.kron(I3,Hd.conj().T)@CZc@np.kron(I3,Hd)):
      M2=np.kron(s,I3)@U1@Xv@U1@CZc@Xv@U1
      print(np.allclose(M2,C2mt))
