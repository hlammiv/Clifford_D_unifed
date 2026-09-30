import numpy as np
z=np.exp(2j*np.pi/9); w=z**3
H=np.array([[1,1,1],[1,w,w**2],[1,w**2,w]])/np.sqrt(3)
T2=np.diag([1,z,z**2])
c=T2.conj().T@H@np.array([1,0,0])
Lam=np.array([[0,0,w],[1,0,0],[0,1,0]])
print("Lam c = zeta c:",np.allclose(Lam@c,z*c))
P2=np.diag([0,0,1]);I=np.eye(3)
C2L=np.kron(I-P2,I)+np.kron(P2,Lam)
D=np.diag([1,1,z])
for psi in [np.random.randn(3)+1j*np.random.randn(3) for _ in range(3)]:
    out=C2L@np.kron(psi,c); print(np.allclose(out,np.kron(D@psi,c)))
# Lam = X * diag(1,1,w)?
X=np.roll(np.eye(3),1,axis=0)
print("Lam==X S':",np.allclose(Lam,X@np.diag([1,1,w])))
Z=np.diag([1,w,w**2])
paulis=[np.kron(np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Z,b),np.linalg.matrix_power(X,c2)@np.linalg.matrix_power(Z,d)) for a in range(3) for b in range(3) for c2 in range(3) for d in range(3)]
def is_pauli(M):
    for P in paulis:
        t=np.trace(P.conj().T@M)/9
        if abs(abs(t)-1)<1e-9 and np.allclose(M,t*P): return True
    return False
def is_clifford(U): return all(is_pauli(U@P@U.conj().T) for P in paulis[1:])
def level(U,maxl=5):
    if is_pauli(U): return 1
    if is_clifford(U): return 2
    for l in range(3,maxl+1):
        # check U P U^dag in level l-1 for generators
        ok=all(levelcheck(U@P@U.conj().T,l-1) for P in [paulis[9],paulis[3],paulis[1],paulis[27]])
        if ok: return l
    return '>%d'%maxl
def levelcheck(V,l):
    if l==1: return is_pauli(V)
    if l==2: return is_clifford(V)
    return all(levelcheck(V@P@V.conj().T,l-1) for P in [paulis[9],paulis[3],paulis[1],paulis[27]])
print("level C2(Lambda):",level(C2L))
print("level diag(1,1,z)xI:",level(np.kron(D,I)))
print("level T2xI:",level(np.kron(T2,I)))
