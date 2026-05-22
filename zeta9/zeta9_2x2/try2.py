import sys, shutil, os
import numpy as np
import matplotlib.pyplot as plt
import glob
from zeta9_vector import find_roots
import json
import tempfile, subprocess
from pathlib import Path

def MatrixDinC(coeff, f):
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

    x = find_roots.vector_coeffs_to_complex(coeff, f)
    
    outer = np.outer(np.conj(x), x)   # x^T x^*
    D = X01 @ (I - outer)

    return D
#

from sage.all import CyclotomicField, Matrix, vector, identity_matrix, ZZ


def zeta9_field():
    K = CyclotomicField(9)
    z = K.gen()
    return K, z


def coeff6_to_ring_element(coeff6, K=None):
    if K is None:
        K, z = zeta9_field()
    else:
        z = K.gen()
    coeff6 = list(coeff6)
    if len(coeff6) != 6:
        raise ValueError("Expected 6 coefficients")
    return sum(ZZ(coeff6[k]) * (z ** k) for k in range(6))


def vector_coeffs_to_exact_vector(coeffs, f, K=None):
    if K is None:
        K, _ = zeta9_field()
    scale = ZZ(3) ** ZZ(f)
    return vector(K, [coeff6_to_ring_element(c, K) / scale for c in coeffs])


def householder_matrix_exact(coeffs, f, K=None):
    if K is None:
        K, _ = zeta9_field()

    x = vector_coeffs_to_exact_vector(coeffs, f, K)

    X01 = Matrix(K, [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    I = identity_matrix(K, 3)

    outer = Matrix(K, 3, 3, lambda i, j: x[i].conjugate() * x[j])
    D = X01 * (I - outer)
    return D


def MatrixDinZ9(coeffs, f, K=None):
    """
    coeffs: shape (3,6)
    returns D = X01 * (I - x^dagger x) exactly over Q(zeta9)
    """
    return householder_matrix_exact(coeffs, f)
#

def exact_matrix_to_complex(D):
    """
    Convert a Sage exact matrix over Q(zeta9) to a NumPy complex128 array.
    """
    nrows = D.nrows()
    ncols = D.ncols()
    out = np.empty((nrows, ncols), dtype=np.complex128)
    for i in range(nrows):
        for j in range(ncols):
            out[i, j] = complex(D[i, j].n())
    return out

    
def Load(theta, eps="1e-1", f=2, mod=""):

    fcache = f"D/X2_f={f}{mod}_eps={eps}_best.npz"
    d = np.load(fcache)
    
    dt = np.abs(theta-d['theta'])
    l = np.argmin(dt)
    assert(dt[l] < np.pi*1e-14)

    vdist = d["vdist"][l]
    coeff = d["coeff"][l]

    D = MatrixDinC(coeff, f)

    target = np.diag([
        np.exp(-0.5j * theta),
        np.exp( 0.5j * theta),
        1.0 + 0.0j,
    ])

    Delta = D - target
    mdist1 = np.linalg.norm(Delta, "fro")


    DZ9 = MatrixDinZ9(coeff, f)    
    #print(DZ9)

    D = exact_matrix_to_complex(DZ9)    
    Delta = D - target
    mdist2 = np.linalg.norm(Delta, "fro")
    assert(np.abs(mdist1-mdist2)/(mdist1+mdist2) < 1e-10)

    vrow = np.linalg.norm(Delta[0,:])
    vcol = np.linalg.norm(Delta[:,0])

    print(f"Lv={vdist:.3g}, Lm={mdist1:.3g}, Lrow={vrow:.3g}, Lcol={vcol:.3g}")
    
#
            
 
def Test(theta, eps="1e-2", lab=None, f=4, suf="", mpi=16, loud=True):

    if(lab is None):
        lab = eps

    triples_file = f"D/TM2_f={f}_eps={lab}"
    if(not os.path.isfile(triples_file)):
        print(f"No triples file {triples_file} found.")
        sys.exit(1)
    #
    triples_json = triples_file + ".manifest.json"
    if(not os.path.isfile(triples_json)):
        print(f"No manifest file {triples_json} found.")
        sys.exit(1)
    #

    rootdb_prefix = f"D/RM2_f={f}_eps={lab}" + suf + "_local"

    chunk_meta_json = f"D/RM2_f={f}_eps={lab}" + suf + "_triples_chunk_meta.json"
    
    d = np.zeros((3,),dtype=np.complex128)
    d[0] = np.exp(-0.5j*theta)
    d[1] = np.exp(0.5j*theta)
    d[2] = 1

    with tempfile.NamedTemporaryFile(suffix="",delete=True) as tmp:
        np.save(tmp.name+".npy",d[0:2])
        cmd = [
            "mpirun", "-n", str(int(mpi)),
            sys.executable,
            "zeta9_2x2/enumerate_vectors_fit_matrix_2x2_mpi.py",
            "--doubles_file", triples_file,
            "--doubles_json", triples_json,
            "--rootdb_prefix", rootdb_prefix,
            "--f", str(int(f)),
            "--diag_target_npy", tmp.name+".npy",
            "--eps", eps,
            "--output_prefix", "try.output",
            "--doubles_chunk_rows", "200000",
            "--chunk_meta_json", chunk_meta_json
        ]
        if(not loud): cmd.append("--quiet")
        
        env = os.environ.copy()
        repo_root = str(Path("zeta9_vector/fit_vectors_mpi.py").resolve().parent.parent)
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        subprocess.run(cmd, cwd=repo_root, env=env, check=False)

        os.remove(tmp.name+".npy")

        d = np.load("try.output.best_matrix.npz")
        for k in d.keys():
            print(k,d[k])
        #
    #
#


theta = np.pi*0.2


Load(theta)

Test(theta,eps="1e-1",lab="1e-2",f=4)
