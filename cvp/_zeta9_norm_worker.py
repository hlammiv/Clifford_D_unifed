"""cvp/_zeta9_norm_worker.py — persistent norm-equation solver subprocess.

This script is **invoked only by ``cvp.diophantine``** in a long-running
sage-env Python subprocess. It loads zeta9's PARI machinery once and then
serves single-norm-equation queries over stdin/stdout JSON.

Wire protocol
-------------

* One JSON object per line on stdin. Two request types:
  ``{"op": "solve", "Y": [m0, m1, m2]}``
  ``{"op": "ping"}``
  ``{"op": "quit"}``
* One JSON object per line on stdout:
  ``{"ok": true, "roots": [[c0,...,c5], ...]}``  (for ``solve``, ``ping``)
  ``{"ok": false, "err": "..."}``                  (on any exception)
* The first stdout line is always ``{"ok": true, "ready": true, "..."}``
  emitted after import + ``build_fields()``.

This is invoked by ``diophantine._NormEqWorker`` and not intended to be
launched manually. Run with::

    PATH=$SAGE_ENV/bin:$PATH $SAGE_ENV/bin/python -m cvp._zeta9_norm_worker

Design rationale
----------------

zeta9's PARI bnfinit (~150 ms) and the full ``build_fields`` (~250 ms)
should be paid once per pipeline call, not once per ``solve_q_norm``
inner loop iteration. Restarting the subprocess between Phase 3 calls
would amortise poorly when ``solve_x2_x3`` issues a few hundred queries.
Persistent worker keeps amortised wall < 5 ms / query.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

# zeta9/ lives next to unified/cvp/; the package import path mirrors
# what zeta9_compile.py uses.
_ZETA9_PKG_PATH = "/home/hlamm/Desktop/efficent_gates/unified/zeta9"
if _ZETA9_PKG_PATH not in sys.path:
    sys.path.insert(0, _ZETA9_PKG_PATH)


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _main():
    try:
        t0 = time.time()
        from zeta9.find_roots_exact_v2 import solve_single_norm_eq
        from zeta9.roots import build_fields

        # Warm caches: build_fields + a trivial PARI call to load bnfinit.
        build_fields()
        solve_single_norm_eq((1, 0, 0))
        dt = time.time() - t0
        _emit({"ok": True, "ready": True, "startup_seconds": dt})
    except Exception as e:
        _emit({"ok": False, "err": f"startup failed: {e}",
               "trace": traceback.format_exc()})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _emit({"ok": False, "err": f"bad json: {e}"})
            continue

        op = req.get("op", "")
        try:
            if op == "ping":
                _emit({"ok": True, "pong": True})
            elif op == "quit":
                _emit({"ok": True, "bye": True})
                return 0
            elif op == "solve":
                Y = tuple(int(c) for c in req["Y"])
                roots = solve_single_norm_eq(Y)
                _emit({"ok": True, "roots": [list(r) for r in roots]})
            else:
                _emit({"ok": False, "err": f"unknown op: {op!r}"})
        except Exception as e:
            _emit({"ok": False, "err": str(e),
                   "trace": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(_main())
