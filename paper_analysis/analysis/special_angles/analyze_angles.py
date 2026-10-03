import numpy as np, sys
from scipy import stats
from pathlib import Path
import os


def _repo_root(start):
    """Directory holding nick_test/: the repository root, or ../unified next to the paper."""
    for p in [start, *start.parents]:
        if (p / "nick_test").is_dir():
            return p
        if (p / "unified" / "nick_test").is_dir():
            return p / "unified"
    raise FileNotFoundError("cannot locate the repository root (directory containing nick_test/)")


rng=np.random.default_rng(0)
for f in map(int,sys.argv[1:]):
    d=np.load(f'ang_f{f}.npz'); th=d['th']; eps=d['eps']
    keep=np.concatenate([[True],np.diff(th)>1e-12]); th=th[keep]; eps=eps[keep]; n=len(th)
    g=np.diff(np.concatenate([[0],th,[np.pi]])); mean=np.pi/(n+1)
    ks=stats.kstest(th/np.pi,'uniform')
    # compare to linear-decreasing density fit
    T=rng.uniform(0,np.pi,20000); idx=np.searchsorted(th,T); best=np.full(T.shape,np.inf)
    for k in (idx-1,idx):
        k=np.clip(k,0,n-1); b=eps[k]+2*np.sqrt(2)*np.abs(np.sin(np.abs(T-th[k])/4)); best=np.minimum(best,b)
    h,_=np.histogram(th,bins=10,range=(0,np.pi))
    print(f"f={f} distinct={n} eps_med={np.median(eps):.2e}  KS D={ks.statistic:.3f} p={ks.pvalue:.1e}  gap CV={g.std()/g.mean():.2f} max/mean={g.max()/mean:.1f} (Poisson~{np.log(n):.1f})")
    print(f"   rand-theta via nearest special: median {np.median(best):.2e}, 90% {np.quantile(best,.9):.2e}; needed density for eps: ~{2*np.sqrt(2)/np.median(eps)/4*np.pi:.1e} angles")
    print("   hist10 (0..pi):",h.tolist())
nt=np.loadtxt(_repo_root(Path(__file__).resolve())/'nick_test'/'fits_f=12.txt',delimiter=',',usecols=0,comments='#')
print('nick_test f=12 thetas:',len(nt),'unique',len(np.unique(nt)),'quantiles',np.quantile(nt,[0,.25,.5,.75,1]).round(3))
