import sys, shutil, os
import numpy as np
import matplotlib.pyplot as plt
import glob
from zeta9.tools import vector_coeffs_to_complex
from zeta9.search_diagonal_matrix_two_rows_streamed_mpi import parse_fixed_row1_sage
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

    x = vector_coeffs_to_complex(coeff, f)
    
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

    
def Load(theta, eps="1e-1", f=2, mod="", loud=False):

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

    mmax = np.max(np.abs(Delta.ravel()))

    DZ9 = MatrixDinZ9(coeff, f)    

    D = exact_matrix_to_complex(DZ9)    
    Delta = D - target
    mdist2 = np.linalg.norm(Delta, "fro")
    assert(np.abs(mdist1-mdist2)/(mdist1+mdist2) < 1e-10)

    vrow = np.linalg.norm(Delta[0,:])
    vcol = np.linalg.norm(Delta[:,0])

    print(f"Lv={vdist:.3g}, Lm={mdist1:.3g}, Lrow={vrow:.3g}, Lcol={vcol:.3g}, Lmax={mmax:.3g}")

    if(loud):
        for i in range(3):
            coeff = parse_fixed_row1_sage(DZ9[i,:].str(),f*2)
            for j in range(3):
                print(f"D[{i},{j}]={coeff[j,:]}, err={np.abs(Delta[i,j]):g}")
            #
        #
    #
    
    return DZ9[0,:].str(), DZ9[1,:].str()
#
            
 
def Test(theta, row1=None, row2=None, eps="1e-1", lab=None, f=2, suf="", mpi=16, loud=True):

    if(lab is None):
        lab = eps

    triples_file = f"D/TM_f={f}_eps={lab}"
    if(not os.path.isfile(triples_file)):
        print(f"No triples file {triples_file} found.")
        sys.exit(1)
    #
    triples_json = triples_file + ".manifest.json"
    if(not os.path.isfile(triples_json)):
        print(f"No manifest file {triples_json} found.")
        sys.exit(1)
    #

    rootdb_prefix = f"D/RM_f={f}_eps={lab}" + suf + "_local"

    chunk_meta_json = f"D/RM_f={f}_eps={lab}" + suf + "_triples_chunk_meta.json"
    
    cmd = [
        "mpirun", "-n", str(int(mpi)),
        sys.executable,
        "zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py",
        "--triples_file", triples_file,
        "--triples_json", triples_json,
        "--rootdb_prefix", rootdb_prefix,
        "--f", str(int(f)),
        "--theta", str(theta),
        "--eps", eps,
        "--output_prefix", "xout",
        "--row2_cache", "xout.row2.npy",
        "--chunk_meta_json", chunk_meta_json,
        #"--disable_y_pruning"
    ]
    if(not loud): cmd.append("--quiet")

    if(row1):
        cmd.append("--fixed_row1_sage")
        cmd.append(row1)
    #
        
    if(row2):
        cmd.append("--fixed_row2_sage")
        cmd.append(row2)
    #

    env = os.environ.copy()
    repo_root = str(Path("zeta9/tools.py").resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    subprocess.run(cmd, cwd=repo_root, env=env, check=False)

    #d = np.load("xout.npz")
    #if(d['frobenius_dist'].size > 0):
    #    print(f"best-fit={np.min(d['frobenius_dist']):g}")
    #
#



theta = np.pi*0.2


row1, row2 = Load(theta)

Test(theta,eps="1e-2",lab="2.5e-3",f=4,mpi=32,row1=row1)
