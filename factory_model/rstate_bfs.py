import numpy as np
w=np.exp(2j*np.pi/3); z=np.exp(2j*np.pi/9)
H=np.array([[1,1,1],[1,w,w*w],[1,w*w,w]])/np.sqrt(3)
S=np.diag([1,1,w]); T=np.diag([1,z,z**8])
def key(M):
    v=M.flatten(); i=np.argmax(np.abs(v)>1e-9); M=M*np.conj(v[i])/abs(v[i])
    return tuple(np.round(M.flatten(),6))
# Clifford group mod phase
cl={key(np.eye(3)):np.eye(3)}; fr=[np.eye(3)]
while fr:
    nf=[]
    for M in fr:
        for G in (H,S):
            N=G@M; kk=key(N)
            if kk not in cl: cl[kk]=N; nf.append(N)
    fr=nf
print("Clifford mod phase:",len(cl))
C=list(cl.values())
def skey(v):
    i=np.argmax(np.abs(v)>1e-9); v=v*np.conj(v[i])/abs(v[i]); return tuple(np.round(v,6))
R=np.array([1,1,-1])/np.sqrt(3)
lvl={skey(c[:,0]):c[:,0] for c in C}
seen=dict(lvl)
for t in range(1,6):
    new={}
    for v in lvl.values():
        u=T@v
        for c in C:
            x=c@u; kk=skey(x)
            if kk not in seen: seen[kk]=x; new[kk]=x
    lvl=new
    best=max(abs(np.vdot(R,x)) for x in new.values()) if new else 0
    print("T-count",t,"new states",len(new),"max |<R|psi>|",round(best,8))
