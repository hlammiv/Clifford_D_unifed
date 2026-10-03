import tarfile, numpy as np, sys, json
from pathlib import Path
import os
z = np.exp(2j*np.pi/9); basis = z**np.arange(6)
# Nick Gnedin's full special-angle output (265 MB, not in the repository; available on request)
TAR = os.environ.get('FITS_TAR', 'fits.tar')
out = {}
fs = [int(a) for a in sys.argv[1:]] or [4,6,8,10,12,14,16]
with tarfile.open(TAR) as tf:
    for f in fs:
        fh = tf.extractfile(f'fits_f={f}.txt')
        best = {}
        nrows = 0
        for raw in fh:
            if raw[:1] == b'#': continue
            nrows += 1
            s = raw.decode()
            th, rest = s.split(',', 1)
            th = float(th)
            groups = [g.split() for g in rest.split(',') if g.strip()][:9]
            a = np.array(groups, dtype=float)
            M = (a @ basis).reshape(3, 3) / 3**f
            R = np.diag([np.exp(-0.5j*th), np.exp(0.5j*th), 1])
            e = np.linalg.norm(M - R)
            if th not in best or e < best[th][0]:
                best[th] = (e, M)
        ths = np.array(sorted(best)); eps = np.array([best[t][0] for t in ths])
        np.savez(f'ang_f{f}.npz', th=ths, eps=eps)
        print(f, 'rows', nrows, 'unique theta', len(ths), 'range', ths.min(), ths.max(),
              'eps median', np.median(eps), 'eps min/max', eps.min(), eps.max(), flush=True)
