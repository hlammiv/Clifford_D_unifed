"""Meet-in-the-middle search: is R=diag(1,1,-1) (up to global phase) a word of
length <= 2L in syllables H*diag(z^a0,z^a1,z^a2)  (unsigned 9th-root diagonals,
which with H generate unsigned single-qutrit Clifford+D)?"""
import numpy as np, itertools, sys
z=np.exp(2j*np.pi/9); w=z**3
H=np.array([[w**(j*k) for k in range(3)] for j in range(3)])/(1j*np.sqrt(3))
syl=[]
for a1 in range(9):
  for a2 in range(9):       # a0=0 WLOG (global phase)
    syl.append(H@np.diag([1,z**a1,z**a2]))
syl=np.array(syl)            # 81 syllables; X^delta is H^2-generated Clifford, covered by longer words
def key(M):
    f=M.flatten(); i=np.argmax(np.abs(f)>1e-6)  # first nonzero entry
    M=M*np.conj(f[i])/abs(f[i])
    return tuple(np.round(np.concatenate([M.real.ravel(),M.imag.ravel()]),6))
R=np.diag([1,1,-1]).astype(complex)
L=int(sys.argv[1]) if len(sys.argv)>1 else 2
# words of length exactly 0..L (left halves)
left={key(np.eye(3)):()}
layer={():np.eye(3,dtype=complex)}
for l in range(L):
    new={}
    for w_,M in layer.items():
        for s in range(81):
            N=M@syl[s]; k=key(N)
            if k not in left: left[k]=w_+(s,); new[w_+(s,)]=N
    layer=new
    print('left size after length',l+1,len(left),flush=True)
# right halves: need R = Wl * Wr  => Wl = R * Wr^{-1}; enumerate Wr over same set
hits=0
Ms={}
for k,w_ in left.items():
    M=np.eye(3,dtype=complex)
    for s in w_: M=M@syl[s]
    kk=key(R@np.linalg.inv(M))
    if kk in left: print('FOUND: R = W1*W2 with W1',left[kk],'W2',w_); hits+=1; break
print('hits',hits,'(covers words up to length',2*L,')')
