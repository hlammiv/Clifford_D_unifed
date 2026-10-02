"""Drop-in launcher for nick_request/analyze_topk.py using fast_decompose.

Same CLI as analyze_topk.py, e.g.
  python3 analyze_topk_fast.py ../nick_request/topk_run/samples/fits_topk_f=8.sample30.txt \
      --k 100 --procs 2 --colmajor --out ../nick_request/topk_run/fast_f8

canonical_reducer.decompose_canonical is replaced by fast_decompose.decompose_fast
before the worker pool forks, so every worker uses the fast path.
"""
import multiprocessing as mp
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
U = D.parent
sys.path[:0] = [str(D), str(U / "nick_test"), str(U / "nick_request"), str(U), str(U / "hrsa")]

import fast_decompose  # noqa: E402

fast_decompose.install()
mp.set_start_method("fork", force=True)   # workers inherit the patched module

import analyze_topk  # noqa: E402

if __name__ == "__main__":
    analyze_topk.main()
