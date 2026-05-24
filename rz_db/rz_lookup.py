"""rz_lookup.py — SQLite-backed lookup table for R_z(theta) approximations.

Phase A of the SK backend project. Wraps the per-cell JSON outputs of the
zeta9 / HRSA sweeps so future SK net construction can resolve an arbitrary
target angle to a stored unitary V (and its decomposition data) in O(log N)
without re-running HRSA/zeta9.

Schema:
    CREATE TABLE rz_entries (
        theta REAL NOT NULL,
        eps_target REAL NOT NULL,
        achieved_frob REAL NOT NULL,
        v_f INTEGER NOT NULL,
        N_D INTEGER,
        method TEXT NOT NULL,
        source TEXT NOT NULL,
        V_blob BLOB NOT NULL,  -- 3*3*6 = 54 int64s in C order, np.tobytes()
        PRIMARY KEY (theta, eps_target)
    );
    CREATE INDEX idx_eps_theta ON rz_entries(eps_target, theta);

Lookup contract (see RzLookupDB.lookup): given target (theta, eps_max),
select the stored entry whose total error budget
    |theta - theta*| * THETA_TO_FROB + achieved_frob
is at most eps_max, preferring smallest N_D (tie-break: smallest budget,
then smallest achieved_frob). Entries with eps_target > eps_max are
ignored (their stored V is not certified tight enough). THETA_TO_FROB
is a conservative constant for ||R_z(a) - R_z(b)||_F as a function of
|a - b| on our U(3) embedding; see the THETA_TO_FROB docstring.
"""
from __future__ import annotations

import sqlite3
from bisect import bisect_left
from pathlib import Path

import numpy as np

V_SHAPE = (3, 3, 6)
V_DTYPE = np.int64
V_NBYTES = int(np.prod(V_SHAPE)) * np.dtype(V_DTYPE).itemsize  # 54 * 8 = 432

# Conversion factor between R_z angle offset and Frobenius distance in U(3).
# For R_z(a) = diag(e^{i a/2}, e^{-i a/2}, 1):
#   ||R_z(a) - R_z(b)||_F^2 = 4 sin^2((a-b)/4)
# small-angle approximation -> |a - b| / sqrt(2). We use 1/2 as a conservative
# upper bound for |a - b| <= pi/2 (since 2 sin(x/4) <= x/2 in this range).
THETA_TO_FROB = 0.5


def _pack_V(V: np.ndarray) -> bytes:
    """Serialize a (3,3,6) int matrix to BLOB. Validates shape/dtype."""
    arr = np.asarray(V)
    if arr.shape != V_SHAPE:
        raise ValueError(f"V has shape {arr.shape}, expected {V_SHAPE}")
    arr = arr.astype(V_DTYPE, casting="safe", copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    blob = arr.tobytes()
    assert len(blob) == V_NBYTES, f"blob is {len(blob)} bytes, expected {V_NBYTES}"
    return blob


def _unpack_V(blob: bytes) -> np.ndarray:
    if len(blob) != V_NBYTES:
        raise ValueError(f"V_blob is {len(blob)} bytes, expected {V_NBYTES}")
    return np.frombuffer(blob, dtype=V_DTYPE).reshape(V_SHAPE).copy()


class RzLookupDB:
    """SQLite-backed lookup for stored R_z(theta) approximations."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS rz_entries (
        theta REAL NOT NULL,
        eps_target REAL NOT NULL,
        achieved_frob REAL NOT NULL,
        v_f INTEGER NOT NULL,
        N_D INTEGER,
        method TEXT NOT NULL,
        source TEXT NOT NULL,
        V_blob BLOB NOT NULL,
        PRIMARY KEY (theta, eps_target)
    );
    CREATE INDEX IF NOT EXISTS idx_eps_theta ON rz_entries(eps_target, theta);
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # WAL allows one writer + many concurrent readers without blocking.
        # Required for MPI sweeps that write back via the lazy-fallback path.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        # Cache of theta lists per eps_target for fast nearest-theta bisect.
        # Invalidated on insert(); rebuilt lazily in lookup().
        self._eps_theta_cache: dict[float, list[float]] | None = None

    # ----- writers ---------------------------------------------------------
    def insert(
        self,
        theta: float,
        eps_target: float,
        V: np.ndarray,
        v_f: int,
        achieved_frob: float,
        N_D: int | None,
        method: str,
        source: str,
    ) -> None:
        """Insert or replace an entry keyed by (theta, eps_target).

        V must be a (3, 3, 6) int array.
        """
        blob = _pack_V(V)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO rz_entries
                (theta, eps_target, achieved_frob, v_f, N_D, method, source, V_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(theta),
                float(eps_target),
                float(achieved_frob),
                int(v_f),
                None if N_D is None else int(N_D),
                str(method),
                str(source),
                blob,
            ),
        )
        self._eps_theta_cache = None

    def insert_many(self, rows: list[dict]) -> int:
        """Bulk insert. rows is a list of kwargs dicts matching insert()."""
        n = 0
        for r in rows:
            self.insert(**r)
            n += 1
        self.commit()
        return n

    def commit(self) -> None:
        self.conn.commit()

    # ----- readers ---------------------------------------------------------
    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM rz_entries")
        return int(cur.fetchone()[0])

    def stats(self) -> dict:
        """Return per-eps tier counts, theta gaps, and per-source counts."""
        out: dict = {"total": self.count(), "eps_tiers": {}, "by_source": {}, "by_method": {}}
        # per-eps tier
        cur = self.conn.execute(
            "SELECT eps_target, COUNT(*) as n, MIN(theta) as tmin, MAX(theta) as tmax "
            "FROM rz_entries GROUP BY eps_target ORDER BY eps_target"
        )
        for row in cur.fetchall():
            eps = float(row["eps_target"])
            n = int(row["n"])
            tmin = float(row["tmin"])
            tmax = float(row["tmax"])
            # compute max gap between consecutive thetas within this eps tier
            thetas = [
                float(r[0])
                for r in self.conn.execute(
                    "SELECT theta FROM rz_entries WHERE eps_target=? ORDER BY theta",
                    (eps,),
                ).fetchall()
            ]
            gaps = [thetas[i + 1] - thetas[i] for i in range(len(thetas) - 1)]
            max_gap = max(gaps) if gaps else 0.0
            min_gap = min(gaps) if gaps else 0.0
            out["eps_tiers"][eps] = {
                "n": n,
                "theta_min": tmin,
                "theta_max": tmax,
                "theta_max_gap": max_gap,
                "theta_min_gap": min_gap,
            }
        # by-source
        cur = self.conn.execute("SELECT source, COUNT(*) FROM rz_entries GROUP BY source")
        for row in cur.fetchall():
            out["by_source"][row[0]] = int(row[1])
        # by-method
        cur = self.conn.execute("SELECT method, COUNT(*) FROM rz_entries GROUP BY method")
        for row in cur.fetchall():
            out["by_method"][row[0]] = int(row[1])
        return out

    def _build_eps_theta_cache(self) -> dict[float, list[float]]:
        cache: dict[float, list[float]] = {}
        cur = self.conn.execute(
            "SELECT eps_target, theta FROM rz_entries ORDER BY eps_target, theta"
        )
        for eps, theta in cur.fetchall():
            cache.setdefault(float(eps), []).append(float(theta))
        return cache

    def lookup(self, theta: float, eps_max: float,
               tightness_first: bool = False) -> dict | None:
        """Return the best stored entry usable as an R_z(theta) approx within eps_max.

        For each eps_target tier <= eps_max, finds the stored theta* closest to
        the target; checks that
            |theta - theta*| * THETA_TO_FROB + achieved_frob <= eps_max

        Ranking:
        - tightness_first=False (default): min N_D, then smallest budget,
          then smallest achieved_frob. Best for paper-data / minimum-cost
          synthesis where any valid cell counts equally.
        - tightness_first=True: min budget (≈ tightest actual approximation),
          then min achieved_frob, then min N_D. Best for SK recursion where
          the Euler approximation precision dominates contraction quality.

        Returns dict with keys: theta, eps_target, achieved_frob, N_D, V,
            v_f, method, source, budget. Returns None if no entry fits.
        """
        if self._eps_theta_cache is None:
            self._eps_theta_cache = self._build_eps_theta_cache()

        candidates: list[tuple] = []
        for eps_t, thetas in self._eps_theta_cache.items():
            if eps_t > eps_max + 1e-15:
                continue
            idx = bisect_left(thetas, theta)
            for cand_idx in (idx - 1, idx):
                if 0 <= cand_idx < len(thetas):
                    theta_star = thetas[cand_idx]
                    d_theta = abs(theta - theta_star)
                    candidates.append((eps_t, theta_star, d_theta))

        if not candidates:
            return None

        best = None
        for eps_t, theta_star, d_theta in candidates:
            row = self.conn.execute(
                "SELECT * FROM rz_entries WHERE eps_target=? AND theta=?",
                (eps_t, theta_star),
            ).fetchone()
            if row is None:
                continue
            budget = d_theta * THETA_TO_FROB + float(row["achieved_frob"])
            if budget > eps_max + 1e-15:
                continue
            n_d = row["N_D"]
            n_d_key = float("inf") if n_d is None else int(n_d)
            if tightness_first:
                # Tighten first: pick the cell with smallest budget (= tightest
                # actual approximation accounting for θ-interpolation distance).
                # Tie-break by achieved_frob then N_D.
                key = (budget, float(row["achieved_frob"]), n_d_key)
            else:
                # Cheap first: min N_D, then budget, then achieved_frob.
                key = (n_d_key, budget, float(row["achieved_frob"]))
            if best is None or key < best[0]:
                best = (key, row, budget)

        if best is None:
            return None
        _, row, budget = best
        return {
            "theta": float(row["theta"]),
            "eps_target": float(row["eps_target"]),
            "achieved_frob": float(row["achieved_frob"]),
            "v_f": int(row["v_f"]),
            "N_D": None if row["N_D"] is None else int(row["N_D"]),
            "method": str(row["method"]),
            "source": str(row["source"]),
            "V": _unpack_V(row["V_blob"]),
            "budget": budget,
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.commit()
        self.close()
