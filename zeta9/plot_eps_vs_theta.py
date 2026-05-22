import sys, shutil, os
import numpy as np
import matplotlib.pyplot as plt
import glob
from zeta9.tools import vector_coeffs_to_complex
import json
import tempfile, subprocess
from pathlib import Path


def Find(theta, eps="1e-1", f=2, norm=2, mod="", mpi=None, loud=False, mpimax=60):

    if(mpi is None):
        npr = (len(theta)+mpimax-1)//mpimax
        assert(npr > 0)
        mpi = (len(theta)+npr-1)//npr
    #
    if(loud):
        print(f"Fitting {len(theta)} targets on {mpi} ranks...")
    #
    
    verr = np.zeros_like(theta,dtype=float)
    coef = np.zeros((len(theta),3,6),dtype=np.int64)

    triples_file = f"D/T{norm}_f={f}_eps={eps}"
    if(not os.path.isfile(triples_file)):
        print(f"No triples file {triples_file} found.")
        sys.exit(1)
    #
    triples_json = triples_file + ".json"
    if(not os.path.isfile(triples_json)):
        triples_json = triples_file + ".manifest.json"
    #
    if(not os.path.isfile(triples_json)):
        print(f"No manifest file {triples_json} found.")
        sys.exit(1)
    #

    rootdb_prefix = f"D/R{norm}_f={f}{mod}_eps={eps}_local"

    scale = (norm/2)**0.5
    d = np.zeros((len(theta),3),dtype=np.complex128)
    d[:,0] = np.exp(-0.5j*theta)*scale
    d[:,1] = -1*scale
    d[:,2] = 0

    with tempfile.NamedTemporaryFile(suffix="",delete=True) as tmp:
        np.save(tmp.name+".npy",d)
        cmd = [
            "mpirun", "-n", str(int(mpi)),
            sys.executable,
            "zeta9/fit_vectors_mpi_sidecar_binned.py",
            "--f", str(int(f)),
            "--triples_file", triples_file,
            "--triples_json", triples_json,
            "--rootdb_prefix", rootdb_prefix,
            "--targets_npy", tmp.name+".npy",
            "--eps", eps,
            "--output_prefix", tmp.name
        ]
        if(not loud): cmd.append("--quiet")

        env = os.environ.copy()
        repo_root = str(Path("zeta9/tools.py").resolve().parent.parent)
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        subprocess.run(cmd, cwd=repo_root, env=env, check=False)

        os.remove(tmp.name+".npy")
        os.remove(tmp.name+".json")
 
        assert(os.path.isfile(tmp.name+".npz"))
        w = np.load(tmp.name+".npz")
        os.remove(tmp.name+".npz")   

        return w['best_dist'], w['best_u_coeffs']
    #
#


def Load(theta, eps="1e-1", f=2, norm=2, mod="", mpi=None, loud=False):

    fcache = f"D/X{norm}_f={f}{mod}_eps={eps}_best.npz"
    try:
        d = np.load(fcache)
        if(theta is None): theta = d["theta"]
    except:
        d = None
        assert(not theta is None)
    #

    vdist = np.zeros_like(theta)
    coeff = np.zeros((len(theta),3,6),dtype=np.int64)

    if(d):
        missing = []
        for i, theta1 in enumerate(theta):
            dt = np.abs(theta1-d['theta'])
            l = np.argmin(dt)
            if(dt[l] > np.pi*1e-14):
                missing.append(i)
            else:
                vdist[i] = d["vdist"][l]
                coeff[i] = d["coeff"][l]
            #
        #
    else:
        missing = list(range(len(theta)))
    #

    if(missing):
        v1, c1 = Find(theta[missing],f=f,norm=norm,eps=eps,mod=mod,mpi=mpi,loud=loud)
        vdist[missing] = v1[:]
        coeff[missing] = c1[:]
        np.savez(fcache,theta=theta,vdist=vdist,coeff=coeff)
    #

    return vdist, coeff
#


def MatNorm(theta, coeff, scale, f):
    """
    D = X_(0,1) * (I - x^T x^*)
    where X_(0,1) swaps the first two basis vectors:
      [[0,1,0],[1,0,0],[0,0,1]]
    and x^T x^* is the outer product x_i conj(x_j).
    """
    X01 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.complex128)
    I = np.eye(3, dtype=np.complex128)

    mdist = np.zeros_like(theta)

    for i, coef1 in enumerate(coeff):
        if(not np.all(coef1 == 0)):
            x = vector_coeffs_to_complex(coef1, f)
    
            outer = scale*np.outer(np.conj(x), x)   # x^T x^*
            D = X01 @ (I - outer)

            target = np.diag([
                np.exp(-0.5j * theta[i]),
                np.exp( 0.5j * theta[i]),
                1.0 + 0.0j,
            ])

            Delta = D - target
            mdist[i] = np.linalg.norm(Delta, "fro")
        else:
            mdist[i] = np.nan
        #
    #

    return mdist
#

    
def Plot(ax, theta=None, eps="1e-1", f=2, norm=2, mod="", color="k", ms=6, mpi=None, label=None,  loud=True):

    theta = np.pi*np.arange(0.1,1.001,0.1)
    #theta = np.pi*np.array([0.2])

    vdist, coeff = Load(theta,f=f,norm=norm,eps=eps,mod=mod,mpi=mpi,loud=loud)

    if(loud):
        n_nan = 0
        n_bad = 0
        scale = (norm/2)**0.5
        d = np.zeros((len(theta),3),dtype=np.complex128)
        d[:,0] = np.exp(-0.5j*theta)*scale
        d[:,1] = -1*scale
        d[:,2] = 0

        for i, coef1 in enumerate(coeff):
            if(vdist[i]):
                x = vector_coeffs_to_complex(coef1, f)
                if(np.max(np.abs(d[i]-x)) > float(eps)): n_bad += 1
            else:
                n_nan += 1
            #
        #
        print(f"N={norm}, f={f}, eps={eps}, max={np.nanmax(vdist):g}, missing={n_nan}, unconfirmed={n_bad}") 
    #
    
    #mdist = MatNorm(theta, coeff, 2/norm, f)

    ax.plot(theta/np.pi,vdist,marker="s",ms=ms,lw=0,color=color,label=label)
#



fig, ax = plt.subplots(figsize=(9,7))

ax.set_xlabel(r"$\theta/\pi$")
ax.set_xlim(0,1)

ax.set_ylabel(r"$L_\min^V$")
ax.set_yscale("log")


#Plot(ax,eps="1e-1",f=2,color="C0",label=r"$|u|^2=2, f_{V}=2$")
#Plot(ax,eps="1e-3",f=3,color="C1",label=r"$|u|^2=2, f_{V}=3$")
#Plot(ax,eps="3e-3",f=3,norm=1,color="C2",label=r"$|u|^2=1, f_{V}=3$")
#Plot(ax,eps="4e-5",f=4,color="b",ms=12,label=r"$|u|^2=2, f_{V}=4$")
#Plot(ax,eps="6e-5",f=4,color="r",label=r"$|u|^2=2, f_{V}=4$")

Plot(ax,eps="1e-3",f=3,color="b",ms=12,label=r"$|u|^2=2, f_{V}=3$")
Plot(ax,eps="1e-3",f=3,color="r",mod="_alt",mpi=10,label=r"$|u|^2=2, f_{V}=3$")


#Plot(ax,eps="6e-5",f=4,color="b",ms=12,label=r"$|u|^2=2, f_{V}=4$")
#Plot(ax,eps="6e-5",f=4,color="r",mod="_alt",mpi=10,label=r"$|u|^2=2, f_{V}=4$")

ax.legend(loc=1)
fig.tight_layout()
plt.show()


